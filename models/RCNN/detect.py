import torch
import cv2
import numpy as np
import sys
from torchvision import transforms
from torchvision.ops import nms

from model import RCNN
from utils import visualize_results, selective_search, parse_yaml_config


def predict(model, image, device, **kwargs):
    num_classes = kwargs['num_classes']
    confidence_thresh = kwargs.get('confidence_thresh', 0.5)
    nms_thresh = kwargs.get('nms_thresh', 0.3)
    # --- 优化点 4: 增加 mini-batch size 参数 ---
    batch_size = kwargs.get('batch_size', 128)

    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 1. 生成候选区域 (这是主要的速度瓶颈)
    proposals = np.array([[x, y, x + w, y + h] for x, y, w, h in selective_search(image)[:2000]])

    final_boxes, final_scores, final_labels = [], [], []

    # --- 优化点 4: 对所有候选区域进行分批处理 ---
    for i in range(0, len(proposals), batch_size):
        batch_proposals = proposals[i:i + batch_size]
        batch_images, valid_proposals = [], []

        # 2. 预处理当前批次的 RoIs
        for box in batch_proposals:
            roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                batch_images.append(transform(roi))
                valid_proposals.append(box)

        if not batch_images:
            continue

        input_tensor = torch.stack(batch_images).to(device)
        current_proposals = np.array(valid_proposals)

        # 3. 模型推理
        with torch.no_grad():
            class_scores, bbox_deltas = model(input_tensor)

        scores = torch.softmax(class_scores, dim=1)[:, 1:]  # 忽略背景类别
        bbox_deltas = bbox_deltas.view(-1, num_classes, 4)

        # 4. 后处理 (解码、NMS)
        for class_idx in range(num_classes):
            class_scores_i = scores[:, class_idx]
            keep_indices = torch.where(class_scores_i > confidence_thresh)[0]

            if len(keep_indices) == 0:
                continue

            # 筛选出符合条件的 proposals, scores, 和 deltas
            boxes_i = torch.from_numpy(current_proposals).to(device)[keep_indices].float()
            scores_i = class_scores_i[keep_indices]
            deltas_i = bbox_deltas[keep_indices, class_idx, :]

            # BBox 回归解码
            p_w, p_h = boxes_i[:, 2] - boxes_i[:, 0], boxes_i[:, 3] - boxes_i[:, 1]
            p_x, p_y = boxes_i[:, 0] + p_w / 2, boxes_i[:, 1] + p_h / 2
            t_x, t_y, t_w, t_h = deltas_i.T
            g_x, g_y = p_w * t_x + p_x, p_h * t_y + p_y
            g_w, g_h = p_w * torch.exp(t_w), p_h * torch.exp(t_h)
            refined_boxes = torch.stack([g_x - g_w / 2, g_y - g_h / 2, g_x + g_w / 2, g_y + g_h / 2], dim=1)

            # --- 优化点 3: 使用 torchvision.ops.nms ---
            keep = nms(refined_boxes, scores_i, nms_thresh)

            final_boxes.extend(refined_boxes[keep].cpu().numpy().tolist())
            final_scores.extend(scores_i[keep].cpu().numpy().tolist())
            final_labels.extend([class_idx] * len(keep))

    return final_boxes, final_scores, final_labels


def run_detection(**kwargs):
    yaml_path = kwargs['yaml_path']
    model_path = kwargs['model_path']
    image_path = kwargs['image_path']

    yaml_config = parse_yaml_config(yaml_path)
    if not yaml_config:
        sys.exit("YAML config could not be parsed. Exiting.")

    class_names = yaml_config['names']
    num_classes = int(yaml_config['nc'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN(num_classes=num_classes).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return
    print("Model loaded successfully.")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predict_params = {
        'num_classes': num_classes,
        'confidence_thresh': kwargs.get('confidence_thresh', 0.5),
        'nms_thresh': kwargs.get('nms_thresh', 0.3),
        'batch_size': kwargs.get('batch_size', 128)
    }
    boxes, scores, labels = predict(model, image_rgb, device, **predict_params)
    print(f"Detected {len(boxes)} objects.")

    vis_params = {'image': image, 'boxes': boxes, 'labels': labels, 'scores': scores, 'class_names': class_names}
    visualize_results(**vis_params)


if __name__ == '__main__':
    # --- 优化点 1: 移除 argparse，改用字典配置 ---
    detection_config = {
        "image_path": "E:/Projects/Datasets/example/brown_blight2.png",
        "yaml_path": "E:/Projects/Datasets/football-players-detection/data.yaml",
        "model_path": "./runs/train4/best.pth",
        "confidence_thresh": 0.5,  # 可以适当调低，因为NMS会过滤掉很多
        "nms_thresh": 0.3,
        "batch_size": 128  # 推理时的批处理大小
    }
    run_detection(**detection_config)