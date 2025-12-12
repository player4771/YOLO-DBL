import cv2
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.tv_tensors import BoundingBoxes

from .tools import rand_rgb

__all__ = (
    'YOLODataset',
    'label_image',
    'label_image_tea',
)

class YOLODataset(torch.utils.data.Dataset):
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

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_raw, w_raw, _ = image.shape  # height, width, channels

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
                x_min = (x_center - w / 2) * w
                y_min = (y_center - h / 2) * h
                x_max = (x_center + w / 2) * w
                y_max = (y_center + h / 2) * h

                boxes.append([x_min, y_min, x_max, y_max])
                # 注意：YOLO类别索引从0开始，SSD的有效类别从1开始（0是背景），所以要+1
                labels.append(int(class_id) + 1)

        w_new, h_new = w_raw, h_raw
        if self.transform:
            transformed = self.transform(image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            _, h_new, w_new = image.shape #通道顺序会发生变化，详见transforms.py

        if boxes:# 转换为 Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=(h_new, w_new))
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
            'orig_size': torch.tensor([h_raw, w_raw]),
            'area': area,
            'iscrowd': iscrowd,
        }

        return image, target

    def get_targets(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[img_path.stem]

        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError(f"Invalid label format in {label_path}, got {len(parts)}/5 parts")
                    class_id, x_c, y_c, w, h = map(float, parts)

                    if self.transform is not None: #这里是为了让box的分辨率与resize图像匹配
                        w, h = self.transform.resize_w, self.transform.resize_h
                    else:
                        with Image.open(img_path) as img:
                            w, h = img.size

                    x_min = (x_c - w / 2) * w
                    y_min = (y_c - h / 2) * h
                    x_max = (x_c + w / 2) * w
                    y_max = (y_c + h / 2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id) + 1)

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        return {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([h, w]),
            'area': area,
            'iscrowd': iscrowd,
        }

def label_image(img_file:str|Path, label_file:str|Path=None, class_names:tuple|list=None, colors:tuple|list=None):
    if label_file is None:
        #YOLO格式, image dir -> label dir -> label file
        label_file = Path(img_file, "../../labels") / (Path(img_file).stem+'.txt')

    image = cv2.imread(img_file)
    h, w, c = image.shape

    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = tuple(map(float, label.strip().split())) #tuple[float*5]
        class_id = int(parts[0])

        # 坐标反归一化 (转为绝对像素)
        x_center = int(parts[1] * w)
        y_center = int(parts[2] * h)
        width = int(parts[3] * w)
        height = int(parts[4] * h)

        # 左上角(x_min, y_min), 右下角(x_max, y_max)
        x_min = int(x_center - (width / 2))
        y_min = int(y_center - (height / 2))
        x_max = x_min + width
        y_max = y_min + height

        text = class_names[class_id] if class_names is not None else f"class {class_id}"
        color = colors[class_id] if colors is not None else rand_rgb() #如果不存在则随机一个
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2) #类别框
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness) #获得文本大小

        if y_min - text_h - baseline < 0: #上方空间不足
            text_y = y_min + text_h + (baseline // 2)
            text_y = min(text_y, y_max - (baseline // 2))
            text_bg_y_min = y_min
            text_bg_y_max = y_min + text_h + baseline
            text_bg_y_max = min(text_bg_y_max, y_max)
        else:
            text_y = y_min - (baseline // 2)
            text_bg_y_min = y_min - text_h - baseline
            text_bg_y_min = max(text_bg_y_min, 0)
            text_bg_y_max = y_min

        cv2.rectangle(image, (x_min, text_bg_y_min), (x_min+text_w, text_bg_y_max), color, -1) #文本背景，-1为填充
        cv2.putText(image, text, (x_min,text_y), font, font_scale, (255, 255, 255), thickness) #文本

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dpi = plt.rcParams['figure.dpi']
    fig,ax = plt.subplots(figsize=(w/dpi,h/dpi), dpi=dpi)
    ax.set_position([0,0,1,1])
    ax.axis('off')
    ax.imshow(image)

    return fig,ax

def label_image_tea(img_file:str|Path=None, label_file:str=None, show:bool=True, save_path:str=None):
    fig, ax = label_image(img_file, label_file,
                       ('algal leaf spot', 'brown blight', 'grey-blight'),
                       ((255,99,71), (165,42,42), (128,128,128)) #orange, brown, grey
                       )
    if show:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, transparent=True)

    return fig, ax