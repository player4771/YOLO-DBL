import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
from pathlib import Path


def get_image_label_paths(img_dir):
    """
    根据图像目录推断标签目录, 并返回图像和标签的文件路径列表。
    假定标签目录与图像目录同级, 且名称为 'labels'。
    例如: .../train/images -> .../train/labels
    """
    img_dir = Path(img_dir)
    # 使用 .parent 导航到上一级目录，然后拼接 'labels'
    label_dir = img_dir.parent / 'labels'

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    # 确保推断出的标签目录存在
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found at the inferred path: {label_dir}")

    label_files = [label_dir / f'{p.stem}.txt' for p in img_files]
    return img_files, label_files


class ObjectDetectionDataset(Dataset):
    def __init__(self, img_paths, label_paths, img_size=(600, 600)):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 主要修改区域开始 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        img_path = str(self.img_paths[idx])  # cv2 需要字符串路径
        label_path = self.label_paths[idx]

        # 1. 使用 OpenCV 读取图像，格式为 BGR, 数据类型为 NumPy array
        img = cv2.imread(img_path)

        # 2. 从 shape 中获取原始高宽
        h, w, _ = img.shape

        # 3. 使用 OpenCV 缩放图像, cv2.resize 接受的尺寸格式是 (宽, 高)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        # 4. 【关键步骤】将颜色空间从 BGR 转换为 RGB
        #    PyTorch 的预训练模型通常使用 RGB 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 5. 将 NumPy 数组转换为 Tensor 并归一化
        #    原有的 self.transform 可以直接处理 NumPy 数组
        img_tensor = self.transform(img)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 主要修改区域结束 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    label = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])

                    x1 = (cx - bw / 2) * self.img_size[0]
                    y1 = (cy - bh / 2) * self.img_size[1]
                    x2 = (cx + bw / 2) * self.img_size[0]
                    y2 = (cy + bh / 2) * self.img_size[1]
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long)

        return img_tensor, boxes, labels, (h, w)


def collate_fn(batch):
    images, bboxes, labels, orig_sizes = zip(*batch)
    return torch.stack(images, 0), list(bboxes), list(labels), list(orig_sizes)


def create_dataloader(img_dir, batch_size, img_size=(600, 600), num_workers=0, shuffle=True):
    """
    创建 DataLoader。
    - num_workers: >0 时开启多进程数据加载。
    - pin_memory=True: 加速数据从 CPU 到 GPU 的传输。
    """
    img_paths, label_paths = get_image_label_paths(img_dir)
    dataset = ObjectDetectionDataset(img_paths, label_paths, img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )