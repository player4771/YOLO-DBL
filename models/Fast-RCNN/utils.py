import torch
import yaml
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from tqdm import tqdm


def parse_data_cfg(path):
    """解析data.yaml文件"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def get_train_proposals_for_dataset(gt_boxes, cfg):
    """为单个图像生成训练用的RoIs(Proposals)"""
    num_pos = int(cfg['roi_num_samples'] * cfg['roi_pos_fraction'])
    if len(gt_boxes) > num_pos:
        perm = torch.randperm(len(gt_boxes))
        positive_rois = gt_boxes[perm[:num_pos]]
    else:
        positive_rois = gt_boxes

    num_neg_candidates = (cfg['roi_num_samples'] - len(positive_rois)) * 5
    random_boxes = torch.zeros((num_neg_candidates, 4), device=gt_boxes.device)

    # 在图像尺寸范围内生成随机候选框
    random_boxes[:, 0] = torch.rand(num_neg_candidates) * cfg['img_size']
    random_boxes[:, 1] = torch.rand(num_neg_candidates) * cfg['img_size']
    random_boxes[:, 2] = torch.rand(num_neg_candidates) * (cfg['img_size'] - random_boxes[:, 0])
    random_boxes[:, 3] = torch.rand(num_neg_candidates) * (cfg['img_size'] - random_boxes[:, 1])
    random_boxes[:, 2:] += random_boxes[:, :2]  # x2,y2 = x1+w, y1+h

    ious = box_iou(random_boxes, gt_boxes)
    max_ious = torch.max(ious, dim=1)[0] if ious.numel() > 0 else torch.zeros(random_boxes.shape[0])

    num_neg = cfg['roi_num_samples'] - len(positive_rois)
    negative_rois = random_boxes[max_ious < cfg['roi_neg_iou_thresh']][:num_neg]

    # 如果负样本不足，允许重复采样以凑够数量
    if len(positive_rois) + len(negative_rois) < cfg['roi_num_samples']:
        if len(negative_rois) > 0:
            shortage = cfg['roi_num_samples'] - (len(positive_rois) + len(negative_rois))
            negative_rois = torch.cat([negative_rois, negative_rois[torch.randint(0, len(negative_rois), (shortage,))]],
                                      dim=0)

    return torch.cat([positive_rois, negative_rois]) if len(positive_rois) > 0 or len(
        negative_rois) > 0 else torch.empty(0, 4)


def compute_targets_for_dataset(proposals, gt_boxes, gt_labels, cfg):
    """为Proposals计算分类标签和回归偏移量"""
    if proposals.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4)

    ious = box_iou(proposals, gt_boxes)
    if ious.numel() == 0:
        target_labels = torch.full((proposals.shape[0],), cfg['num_classes'] - 1, dtype=torch.long)
        target_deltas = torch.zeros_like(proposals)
        return target_labels, target_deltas

    max_ious, max_indices = torch.max(ious, dim=1)
    target_labels = gt_labels[max_indices]
    target_labels[max_ious < cfg['roi_pos_iou_thresh']] = cfg['num_classes'] - 1

    matched_gt_boxes = gt_boxes[max_indices]
    px, py = (proposals[:, 0] + proposals[:, 2]) / 2, (proposals[:, 1] + proposals[:, 3]) / 2
    pw, ph = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6), (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    gx, gy = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2, (
            matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
    gw, gh = (matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]).clamp(min=1e-6), (
            matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]).clamp(min=1e-6)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

    targets = torch.stack((tx, ty, tw, th), dim=1)

    BBOX_REG_MEANS = torch.tensor([0.0, 0.0, 0.0, 0.0])
    BBOX_REG_STDS = torch.tensor([0.1, 0.1, 0.2, 0.2])

    means = BBOX_REG_MEANS.to(targets.device)
    stds = BBOX_REG_STDS.to(targets.device)
    targets = (targets - means) / stds

    return target_labels, targets


class CustomDataset(Dataset):
    def __init__(self, img_path, cfg, is_train=True):
        self.img_path = img_path
        self.img_files = sorted(
            [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.label_files = [f.replace('images', 'labels').replace(os.path.splitext(f)[1], '.txt') for f in
                            self.img_files]
        self.cfg = cfg
        self.is_train = is_train

        # --- 核心修改：缓存逻辑 ---
        # 定义缓存文件路径，例如: .../dataset/train.cache
        parent_dir = os.path.abspath(os.path.join(img_path, os.pardir, os.pardir))
        set_name = os.path.basename(os.path.abspath(os.path.join(img_path, os.pardir)))
        cache_path = os.path.join(parent_dir, f"{set_name}.cache")

        if os.path.exists(cache_path):
            print(f"Loading cache from {cache_path}")
            self.cached_items = torch.load(cache_path)
            if len(self.cached_items) != len(self.img_files):
                print("Cache is outdated or invalid, re-creating...")
                self.cached_items = self._create_cache(cache_path)
        else:
            self.cached_items = self._create_cache(cache_path)

    def _create_cache(self, cache_path):
        """遍历整个数据集，预计算proposals和targets，并保存到文件"""
        set_name = os.path.basename(os.path.abspath(os.path.join(self.img_path, os.pardir)))
        print(f"Creating cache for {set_name} set at {cache_path}...")

        items_to_cache = []
        pbar = tqdm(range(len(self.img_files)), desc=f"Caching {set_name} data")

        for index in pbar:
            # 1. 读取图像尺寸以进行坐标缩放
            img = cv2.imread(self.img_files[index])
            h, w, _ = img.shape
            scale = self.cfg['img_size'] / max(h, w)
            resized_h, resized_w = int(h * scale), int(w * scale)

            # 2. 读取标签
            boxes = []
            label_path = self.label_files[index]
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        cls, x, y, w_box, h_box = map(float, line.split())
                        x1 = (x - w_box / 2) * resized_w
                        y1 = (y - h_box / 2) * resized_h
                        x2 = (x + w_box / 2) * resized_w
                        y2 = (y + h_box / 2) * resized_h
                        boxes.append([int(cls), x1, y1, x2, y2])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            gt_boxes = boxes[:, 1:] if len(boxes) > 0 else torch.empty((0, 4))
            gt_labels = boxes[:, 0].long() if len(boxes) > 0 else torch.empty(0, dtype=torch.long)

            # 3. 生成 proposals 和 targets
            proposals = get_train_proposals_for_dataset(gt_boxes, self.cfg)
            target_labels, target_deltas = compute_targets_for_dataset(proposals, gt_boxes, gt_labels, self.cfg)

            items_to_cache.append((proposals, target_labels, target_deltas))

        torch.save(items_to_cache, cache_path)
        print(f"Cache saved successfully to {cache_path}")
        return items_to_cache

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 图像加载和转换仍然在每次调用时进行
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        scale = self.cfg['img_size'] / max(h, w)
        resized_h, resized_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (resized_w, resized_h))

        new_img = np.zeros((self.cfg['img_size'], self.cfg['img_size'], 3), dtype=np.uint8)
        new_img[0:resized_h, 0:resized_w] = img_resized
        img_tensor = torch.from_numpy(new_img).permute(2, 0, 1).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = normalize(img_tensor)

        # 直接从内存中的缓存加载proposals和targets
        proposals, target_labels, target_deltas = self.cached_items[index]

        return img_tensor, proposals, target_labels, target_deltas


def collate_fn_fastrcnn(batch):
    """为Fast R-CNN专门设计的collate_fn，处理带batch_index的rois"""
    images, proposals, target_labels, target_deltas = zip(*batch)

    images = torch.stack(images, dim=0)

    rois = []
    for i, p in enumerate(proposals):
        if p.numel() > 0:
            rois.append(torch.cat([torch.full((p.shape[0], 1), i, dtype=torch.float), p], dim=1))

    rois = torch.cat(rois, dim=0) if rois else torch.empty(0, 5)
    target_labels = torch.cat(target_labels, dim=0)
    target_deltas = torch.cat(target_deltas, dim=0)

    return images, rois, target_labels, target_deltas


def get_dataloaders(cfg, data_info):
    """创建训练和验证的DataLoader"""
    train_dataset = CustomDataset(img_path=data_info['train'], cfg=cfg, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn_fastrcnn,
                              num_workers=cfg['num_workers'], pin_memory=True)

    val_dataset = CustomDataset(img_path=data_info['val'], cfg=cfg, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn_fastrcnn,
                            num_workers=cfg['num_workers'], pin_memory=True)

    return train_loader, val_loader


def apply_regression(boxes, deltas):
    """
    将模型预测的回归偏移量应用到提议框上。

    Args:
        boxes (torch.Tensor): [N, 4] 格式为 (x1, y1, x2, y2) 的提议框。
        deltas (torch.Tensor): [N, 4] 模型预测的偏移量 (tx, ty, tw, th)。

    Returns:
        torch.Tensor: [N, 4] 调整后的精确边界框。
    """
    # 将提议框从 (x1, y1, x2, y2) 转换为 (cx, cy, w, h)
    px = (boxes[:, 0] + boxes[:, 2]) / 2.0
    py = (boxes[:, 1] + boxes[:, 3]) / 2.0
    pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

    # 提取偏移量
    BBOX_REG_MEANS = torch.tensor([0.0, 0.0, 0.0, 0.0])
    BBOX_REG_STDS = torch.tensor([0.1, 0.1, 0.2, 0.2])

    stds = BBOX_REG_STDS.to(deltas.device)
    means = BBOX_REG_MEANS.to(deltas.device)
    deltas = deltas * stds + means

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 应用逆变换
    pred_cx = px + pw * dx
    pred_cy = py + ph * dy
    pred_w = pw * torch.exp(dw)
    pred_h = ph * torch.exp(dh)

    # 将计算出的新坐标转换回 (x1, y1, x2, y2) 格式
    pred_x1 = pred_cx - pred_w / 2.0
    pred_y1 = pred_cy - pred_h / 2.0
    pred_x2 = pred_cx + pred_w / 2.0
    pred_y2 = pred_cy + pred_h / 2.0

    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)


def convert_and_save_fp16(model_path, num_classes, output_path=None):
    """
    加载一个全精度(float32)模型,将其转换为半精度(float16),并保存.

    Args:
        model_path (str): 输入的 float32 .pth 文件路径.
        num_classes (int): 模型的类别数 (包含背景).
        output_path (str, optional): 输出的 float16 .pth 文件路径.
                                     如果为 None,则在原文件名后添加 "_fp16".
    """
    if not output_path:
        output_path = model_path.replace('.pth', '_fp16.pth')

    print(f"\n--- Starting FP16 Conversion ---")
    print(f"Loading float32 weights from: {model_path}")

    # 导入模型定义,需要确保 model.py 在PYTHONPATH中
    from model import FastRCNN

    # 加载模型结构并置于评估模式
    model = FastRCNN(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 将模型转换为半精度
    print("Converting model to float16...")
    model.half()

    # 保存半精度的 state_dict
    print(f"Saving float16 model to: {output_path}")
    torch.save(model.state_dict(), output_path)
    print("Conversion complete!")