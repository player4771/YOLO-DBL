# detect.py
import torch
import cv2
import numpy as np
import time
import os
from model import FastRCNN
from utils import apply_regression, parse_data_cfg
from torchvision.ops import nms
from torchvision import transforms  # <-- 新增

# --- 新增：为预训练模型定义的归一化 ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def generate_grid_proposals(img_width, img_height, scales=[128, 256, 512], ratios=[0.5, 1, 2], stride=16):
    proposals = []
    for s in scales:
        for r in ratios:
            h = int(np.sqrt(s * s / r))
            w = int(h * r)
            for y in range(0, img_height - h, stride):
                for x in range(0, img_width - w, stride):
                    proposals.append([x, y, x + w, y + h])
    return torch.tensor(proposals, dtype=torch.float32)


def detect(**kwargs):
    # --- 1. 严格的配置管理 ---
    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': 'data/data.yaml',
        'weights': None,  # 修正：添加默认键
        'example': None,  # 修正：添加默认键
        'outfile': 'result.jpg',
        'conf_thres': 0.5,
        'nms_thres': 0.4,
        'img_size': 640,
    }
    cfg.update(kwargs)

    if not cfg['weights'] or not cfg['example']:
        raise ValueError("必须通过参数提供 'weights' 和 'example' 的路径")

    device = torch.device(cfg['device'])
    data_info = parse_data_cfg(cfg['data'])
    num_classes = data_info['nc'] + 1
    class_names = data_info['names']

    # --- 2. 加载模型 ---
    model = FastRCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(cfg['weights'], map_location=device))
    model.to(device).eval()

    # --- 3. 图像预处理 (已修正) ---
    orig_img = cv2.imread(cfg['example'])
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # <-- 修正：BGR 转换为 RGB

    h, w, _ = img_rgb.shape
    scale = cfg['img_size'] / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (resized_w, resized_h))  # 使用转换后的 img_rgb

    # 应用与训练时完全相同的转换
    img_tensor_raw = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor_normalized = normalize(img_tensor_raw)  # <-- 修正：应用归一化

    # 填充到最终尺寸
    img_tensor = torch.zeros((1, 3, cfg['img_size'], cfg['img_size']), device=device)
    img_tensor[0, :, :resized_h, :resized_w] = img_tensor_normalized  # 使用归一化后的 tensor

    print("Generating proposals...")
    proposals = generate_grid_proposals(resized_w, resized_h).to(device)
    rois = torch.cat([torch.zeros(proposals.shape[0], 1, device=device), proposals], dim=1)

    print(f"Running inference on {len(proposals)} proposals...")
    start_time = time.time()
    with torch.no_grad():
        scores, bbox_deltas = model(img_tensor, rois)
    scores = torch.softmax(scores, dim=1)
    print(f"Inference time: {time.time() - start_time:.4f}s")

    final_boxes, final_scores, final_labels = [], [], [],
    bbox_deltas = bbox_deltas.view(-1, num_classes, 4)

    for j in range(num_classes - 1):
        class_scores = scores[:, j]
        keep_indices = (class_scores > cfg['conf_thres']).nonzero(as_tuple=True)[0]
        if len(keep_indices) == 0: continue

        filtered_scores = class_scores[keep_indices]
        filtered_deltas = bbox_deltas[keep_indices, j, :]
        filtered_proposals = proposals[keep_indices]

        refined_boxes = apply_regression(filtered_proposals, filtered_deltas)
        keep = nms(refined_boxes, filtered_scores, cfg['nms_thres'])

        final_boxes.append(refined_boxes[keep])
        final_scores.append(filtered_scores[keep])
        final_labels.append(torch.full_like(filtered_scores[keep], j, dtype=torch.long))

    if final_boxes:
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        final_boxes /= scale
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[label]
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Detection Result', orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result_path = os.path.join(os.path.dirname(cfg['weights']), cfg['outfile'])
    cv2.imwrite(result_path, orig_img)
    print(f"Result saved to {result_path}")


if __name__ == '__main__':
    detect(
        weights="./runs/train4/best.pth",
        example="E:/Projects/Datasets/example/brown_blight2.png",
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        outfile='result.jpg',
        conf_thres=0.1,
        nms_thres=0.5
    )