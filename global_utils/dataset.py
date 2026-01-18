
import torch
import imagesize
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.io import decode_image
from torchvision.tv_tensors import BoundingBoxes

from global_utils.tools import rand_rgb

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

        image = decode_image(str(img_path))
        _, w_raw, h_raw = image.shape #CHW

        # 读取 YOLO 标注
        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid label format in {label_path}, got {len(parts)}/5 parts")

                # YOLO format: class_id(int), x_center, y_center, width, height(normalized, float)
                class_id, x_center, y_center, w, h = map(float, parts)

                # 转换为VOC格式: [xmin, ymin, xmax, ymax]
                x_min = (x_center - w / 2) * w_raw
                y_min = (y_center - h / 2) * h_raw
                x_max = (x_center + w / 2) * w_raw
                y_max = (y_center + h / 2) * h_raw

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id) + 1) #YOLO类别索引从0开始，SSD的有效类别从1开始(0是背景)，所以要+1

        if self.transform: #transform里有标准化，故读取时不进行处理(uint8省空间)
            T = self.transform(image, bboxes=boxes, class_labels=labels) #通道顺序会发生变化，详见transforms.py
            image, boxes, labels = T['image'], T['bboxes'], T['class_labels']
        else:
            image = image.float()/255.0
        _, h_new, w_new = image.shape if self.transform else None, w_raw, h_raw

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
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid label format in {label_path}, got {len(parts)}/5 parts")
                class_id, x_c, y_c, w, h = map(float, parts)

                if self.transform is not None: #这里是为了让box的分辨率与resize图像匹配
                    w_new, h_new = self.transform.resize_w, self.transform.resize_h
                else:
                    w_new, h_new = imagesize.get(img_path)

                x_min = (x_c - w / 2) * w_new
                y_min = (y_c - h / 2) * h_new
                x_max = (x_c + w / 2) * w_new
                y_max = (y_c + h / 2) * h_new

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

def label_image(image_file:str|Path, label_file:str|Path=None, class_names: tuple|list=None, colors: tuple|list=None):
    if label_file is None:
        #YOLO格式, image dir -> label dir -> label file
        label_file = Path(image_file, "../../labels") / (Path(image_file).stem + '.txt')

    image = plt.imread(str(image_file))
    h, w = image.shape[:2]

    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.set_position((0, 0, 1, 1))  # 让Axes充满整个Figure
    ax.axis('off')
    ax.imshow(image)

    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = tuple(map(float, label.strip().split())) #tuple[float*5]
        class_id = int(parts[0])

        #坐标转换 (YOLO Center -> Top-Left)
        box_w = parts[3] * w
        box_h = parts[4] * h
        x_center = parts[1] * w
        y_center = parts[2] * h
        x_min = x_center - (box_w / 2)
        y_min = y_center - (box_h / 2)

        text = class_names[class_id] if class_names is not None else f"class {class_id}"
        color = colors[class_id] if colors is not None else rand_rgb() #如果不存在则随机一个
        if max(color) > 1.0:
            color = tuple([c/255.0 for c in color])

        rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none') #类别框
        ax.add_patch(rect)

        pad = 2
        ax.text(
            x_min+pad, y_min+pad, text, color='white', fontsize=max(10, h//60),
            verticalalignment='top' if y_min<(h*0.05) else 'bottom', #文字是在线上方还是线下方
            bbox={'facecolor': color, 'edgecolor': 'none', 'alpha': 1.0, 'pad': pad}
        )

    return fig,ax

def label_image_tea(image_file:str|Path=None, label_file:str=None, show:bool=True, save_path:str=None):
    fig, ax = label_image(
        image_file, label_file,
        ('algal leaf spot', 'brown blight', 'grey-blight'),
        ((255,99,71), (165,42,42), (128,128,128)) #orange, brown, grey
    )
    if show:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, transparent=True)
    return fig, ax

if __name__ == '__main__':
    label_image_tea(
        image_file=r"E:\Projects\Datasets\example\sample_v4_1.jpg",
        label_file=r"E:\Projects\Datasets\example\sample_v4_1.txt",
    )