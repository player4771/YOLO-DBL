import torch
import cv2
import numpy as np
from torchvision import transforms, ops

from model import RCNN
from utils import visualize_results, selective_search, read_yaml


def predict(model, image, device, **kwargs):
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # --- 性能优化: 使用NumPy向量化操作进行坐标转换 ---
    rects = selective_search(image)[:2000]
    if len(rects) == 0:
        return [], [], []
    # 通过向量化操作将 [x, y, w, h] 转换为 [x1, y1, x2, y2]
    proposals = rects.astype(np.float32)
    proposals[:, 2] += proposals[:, 0]  # x2 = w + x1
    proposals[:, 3] += proposals[:, 1]  # y2 = h + y1

    # --- NMS逻辑修正: 步骤1 - 分批推理并收集所有原始输出 ---
    all_class_scores = []
    all_bbox_deltas = []
    all_proposals = []

    for i in range(0, len(proposals), kwargs['batch_size']):
        batch_proposals = proposals[i:i + kwargs['batch_size']]
        batch_images, valid_proposals = [], []

        for box in batch_proposals:
            x1, y1, x2, y2 = map(int, box)
            if x1 >= x2 or y1 >= y2: continue
            roi = image[y1:y2, x1:x2]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                batch_images.append(transform(roi))
                valid_proposals.append(box)

        if not batch_images:
            continue

        input_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            class_scores, bbox_deltas = model(input_tensor)

        all_class_scores.append(class_scores)
        all_bbox_deltas.append(bbox_deltas)
        all_proposals.extend(valid_proposals)

    if not all_proposals:
        return [], [], []

    all_class_scores = torch.cat(all_class_scores, dim=0)
    all_bbox_deltas = torch.cat(all_bbox_deltas, dim=0)
    all_proposals_tensor = torch.from_numpy(np.array(all_proposals)).to(device)

    # --- NMS逻辑修正: 步骤2 - 对所有结果进行全局后处理 ---
    final_boxes, final_scores, final_labels = [], [], []

    scores = torch.softmax(all_class_scores.float(), dim=1)[:, 1:]
    bbox_deltas_reshaped = all_bbox_deltas.float().view(-1, kwargs['nc'], 4)

    for class_idx in range(kwargs['nc']):
        class_scores_i = scores[:, class_idx]
        keep_indices = torch.where(class_scores_i > kwargs['confidence_thresh'])[0]

        if len(keep_indices) == 0:
            continue

        boxes_i = all_proposals_tensor[keep_indices]
        scores_i = class_scores_i[keep_indices]
        deltas_i = bbox_deltas_reshaped[keep_indices, class_idx, :]

        p_w, p_h = boxes_i[:, 2] - boxes_i[:, 0], boxes_i[:, 3] - boxes_i[:, 1]
        p_x, p_y = boxes_i[:, 0] + p_w / 2, boxes_i[:, 1] + p_h / 2
        t_x, t_y, t_w, t_h = deltas_i.T
        g_x, g_y = p_w * t_x + p_x, p_h * t_y + p_y
        g_w, g_h = p_w * torch.exp(t_w), p_h * torch.exp(t_h)
        refined_boxes = torch.stack([g_x - g_w / 2, g_y - g_h / 2, g_x + g_w / 2, g_y + g_h / 2], dim=1)

        keep = ops.nms(refined_boxes, scores_i, kwargs['nms_thresh'])

        final_boxes.extend(refined_boxes[keep].cpu().numpy().tolist())
        final_scores.extend(scores_i[keep].cpu().numpy().tolist())
        final_labels.extend([class_idx] * len(keep))

    return final_boxes, final_scores, final_labels


def detect(**kwargs):
    config = read_yaml(kwargs['yaml_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN(num_classes=config['nc']).to(device)

    model.load_state_dict(torch.load(kwargs['model_path'], map_location=device))

    print("Model loaded successfully.")

    image = cv2.imread(kwargs['image_path'])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kwargs['nc'] = config['nc']
    boxes, scores, labels = predict(model, image_rgb, device, **kwargs)
    print(f"Detected {len(boxes)} objects.")

    vis_params = {'image': image, 'boxes': boxes, 'labels': labels, 'scores': scores, 'class_names': config['names']}
    visualize_results(**vis_params)


if __name__ == '__main__':
    detection_config = {
        "image_path": "E:/Projects/Datasets/example/brown_blight2.png",
        "yaml_path": "E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        "model_path": "./runs/train4/best.pth",
        "confidence_thresh": 0.5,  # 可以适当调低，因为NMS会过滤掉很多
        "nms_thresh": 0.5,
        "batch_size": 64  # 推理时的批处理大小
    }
    detect(**detection_config)