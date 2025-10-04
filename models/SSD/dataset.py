
import torch
from PIL import Image
from pathlib import Path
from torchvision.tv_tensors import BoundingBoxes

class YoloDataset(torch.utils.data.Dataset):
    """
    用于读取 YOLO 格式数据集的 PyTorch Dataset 类。
    """

    def __init__(self, img_dir, label_dir, transform=None):
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
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # 读取 YOLO 标注
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])

                # 转换为 [xmin, ymin, xmax, ymax] 绝对坐标
                x_min = (x_center - w / 2) * img_w
                y_min = (y_center - h / 2) * img_h
                x_max = (x_center + w / 2) * img_w
                y_max = (y_center + h / 2) * img_h

                boxes.append([x_min, y_min, x_max, y_max])
                # 注意：YOLO类别索引从0开始，SSD的有效类别从1开始（0是背景），所以要+1
                labels.append(class_id + 1)

        # 转换为 Tensor
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            #print('boxes is empty')
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": BoundingBoxes(boxes, format="xyxy", canvas_size=(img_h, img_w)),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([img_h, img_w])
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target