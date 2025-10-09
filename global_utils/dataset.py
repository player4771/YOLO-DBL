import cv2
import torch
from pathlib import Path
from torchvision.tv_tensors import BoundingBoxes

class YoloDataset(torch.utils.data.Dataset):
    """用于读取 YOLO 格式数据集的 PyTorch Dataset 类。"""

    def __init__(self, img_dir:str|Path, label_dir:str|Path, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        # 筛选出有对应标注的图片
        self.img_files = sorted([p for p in self.img_dir.iterdir() if p.suffix in ['.jpg', '.png', '.jpeg']])
        self.label_files = {p.stem: p for p in self.label_dir.iterdir() if p.suffix == '.txt'}

        self.img_files = [p for p in self.img_files if p.stem in self.label_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[img_path.stem]

        # 读取图片
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape # height, width, channels

        # 读取 YOLO 标注
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid label format in {label_path}, got {len(parts)}/5 parts")

                # YOLO format: class_id(int), x_center, y_center, width, height(normalized, float)
                class_id, x_center, y_center, w, h = map(float, parts)

                # 转换为VOC格式: [xmin, ymin, xmax, ymax]
                x_min = (x_center - w / 2) * img_w
                y_min = (y_center - h / 2) * img_h
                x_max = (x_center + w / 2) * img_w
                y_max = (y_center + h / 2) * img_h

                boxes.append([x_min, y_min, x_max, y_max])
                # 注意：YOLO类别索引从0开始，SSD的有效类别从1开始（0是背景），所以要+1
                labels.append(int(class_id) + 1)


        if self.transform:
            transformed = self.transform(image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            _, img_h, img_w = image.shape #通道顺序会发生变化，详见transforms.py

        if boxes:# 转换为 Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=(img_h, img_w))
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([img_h, img_w]),
            'area': area,
            'iscrowd': iscrowd,
        }

        return image, target