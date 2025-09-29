import os
import yaml
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class YOLODataSet(Dataset):
    def __init__(self, yaml_path, mode="train", transforms=None):
        """
        初始化 YOLO 数据集。

        Args:
            yaml_path (str): data.yaml 文件的路径。
            mode (str): 'train', 'val', or 'test'，指定要加载的数据集部分。
            transforms (callable, optional): 应用于图像和目标的可选转换。
        """
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val', or 'test'."
        self.transforms = transforms

        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)

        self.nc = data['nc']
        self.class_names = data['names']

        # 获取 yaml 文件所在的目录作为根目录
        dataset_root = os.path.dirname(yaml_path)
        image_dir = os.path.join(dataset_root, data[mode])

        # 获取所有支持的图像文件
        self.image_paths = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for fname in os.listdir(image_dir):
            if any(fname.lower().endswith(ext) for ext in supported_formats):
                self.image_paths.append(os.path.join(image_dir, fname))

        # --- 3. 获取对应的标签路径列表 ---
        # 标签路径是根据图像路径生成的
        self.label_paths = [
            os.path.join(os.path.dirname(p).replace("images", "labels").__str__(),
                         os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- 1. 读取图像 ---
        img_path = self.image_paths[idx]
        # 使用 OpenCV 读取图像 (BGR格式)
        image = cv2.imread(img_path)
        # 将 BGR 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # --- 2. 读取标签文件 ---
        label_path = self.label_paths[idx]
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # 解析每行: class_id, x_center, y_center, width, height
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_c, y_c, w_norm, h_norm = [float(p) for p in parts]

                    # YOLO 格式 (x_center, y_center, width, height) 转换为 (xmin, ymin, xmax, ymax)
                    # 坐标已经是相对于图像尺寸归一化的
                    xmin = x_c - w_norm / 2
                    ymin = y_c - h_norm / 2
                    xmax = x_c + w_norm / 2
                    ymax = y_c + h_norm / 2

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id))

        # --- 3. 转换为 Torch Tensors ---
        # 注意：这里的 boxes 已经是归一化后的坐标了
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 确保即使没有标注框，也能返回正确形状的空 Tensor
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # --- 4. 构建 target 字典 ---
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            # 计算面积 (归一化坐标下的面积)
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            # YOLO 格式不包含 iscrowd 信息，默认为0
            "iscrowd": torch.zeros_like(labels, dtype=torch.int64),
            # 存储原始图像尺寸
            "height_width": torch.as_tensor([int(h), int(w)])
        }

        # --- 5. 应用数据增强 ---
        if self.transforms is not None:
            # 注意：大部分数据增强库（如Albumentations）接受的图像是NumPy数组，标签格式可能需要适配
            # 这里的 image 是 (H, W, C) 的 NumPy 数组
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        """
        自定义的 collate_fn，用于处理 batch 中样本标注框数量不同的情况。
        返回一个包含图像元组和目标元组的列表。
        """
        images, targets = tuple(zip(*batch))
        return images, targets