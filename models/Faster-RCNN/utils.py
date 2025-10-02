import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
torch.hub.set_dir('./') #修改缓存路径
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection import faster_rcnn
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EarlyStopping:
    """
    当验证集指标在连续 'patience' 个 epoch 内没有提升时，提前终止训练。
    """

    def __init__(self, patience=5, verbose=False, delta=0, path='best.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def update(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = score


def find_new_dir(dir_: str | Path) -> str | Path: #给定默认路径，寻找下一个未被占用的路径
    ret = str(dir_)
    while Path(ret).exists():
        num = re.search(r'\d+$', ret)
        if num: #结尾有数字：序号+1
            ret = ret[:num.start()] + str(int(num.group(0)) + 1)
        else: #结尾没数字：添加序号2
            ret = ret + '2'
    if isinstance(dir_, str): #返回类型与传入类型保持相同，用起来方便
        return str(ret)
    elif isinstance(dir_, Path):
        return Path(ret)
    else:
        raise TypeError('dir_ = str/Path')

def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1

    # This is the dictionary we are fixing
    dataset = {
        "info": {},  # ✅ ADDED: Empty info dictionary
        "licenses": [],  # ✅ ADDED: Empty licenses list
        "images": [],
        "categories": [],
        "annotations": []
    }

    categories = set()
    for img_idx in range(len(ds)):
        # Make sure your dataset returns image, not just shape
        try:
            img, targets = ds[img_idx]
            height, width = img.shape[-2], img.shape[-1]
        except:
            # Fallback for datasets that might not return the full image
            targets = ds.get_target(img_idx)  # Assuming a helper if needed
            height, width = targets.get("height", 512), targets.get("width", 512)

        image_id = targets["image_id"].item()
        img_dict = {
            'id': image_id,
            'height': height,
            'width': width,
        }
        dataset["images"].append(img_dict)

        bboxes = targets["boxes"]
        # Convert from xyxy to xywh
        bboxes[:, 2:] -= bboxes[:, :2]

        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        for i in range(len(bboxes)):
            ann = {
                'image_id': image_id,
                'bbox': bboxes[i].tolist(),
                'category_id': labels[i],
                'area': areas[i],
                'iscrowd': iscrowd[i],
                'id': ann_id,
            }
            categories.add(labels[i])
            dataset["annotations"].append(ann)
            ann_id += 1

    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


@torch.no_grad()
def evaluate(model, data_loader, device, outfile=None):
    model.eval()

    coco_gt = convert_to_coco_api(data_loader.dataset)

    # 使用一个字典来存储所有图片的预测结果，键是 image_id
    results_dict = {}
    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(img.to(device, non_blocking=True) for img in images)

        # 在某些设备上需要同步以获得准确时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # 将这个批次的预测结果更新到总的 results_dict 中
        batch_res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        results_dict.update(batch_res)

    # 准备 COCO 格式的检测结果
    coco_results = []
    # 按照 COCO ground truth 的顺序来整理结果
    for original_id in coco_gt.getImgIds():
        # 如果模型没有对某张图片做出预测（例如，图片中没有识别到任何物体），则跳过
        if original_id not in results_dict:
            continue

        prediction = results_dict[original_id]
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        # 转换成 COCO 的 bbox 格式 (x, y, width, height)
        boxes[:, 2:] -= boxes[:, :2]

        # 扩展到 coco_results 列表
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k].item(),
                    "bbox": boxes[k].tolist(),
                    "score": scores[k].item(),
                }
                for k in range(len(boxes))
            ]
        )

    if not coco_results:
        print("No predictions were made, skipping COCO evaluation.")
        return None

    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if outfile is not None:
        write_coco_stat(coco_eval.stats, outfile)

    return coco_eval

def write_coco_stat(stat:list, outfile:str|Path) -> None:
    metric_names = [
        'mAP', 'AP50', 'AP75',
        'APs', 'APm', 'APl',
        'AR1', 'AR10', 'AR100',
        'ARs', 'ARm', 'ARl'
    ]

    if not Path(outfile).exists():
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        # Dataframe默认接受一个二维列表，第一维为列，内部每个列表为行
        df = pd.DataFrame([stat], columns=metric_names)
    else:
        df = pd.read_csv(outfile)
        if df.empty:
            df = pd.DataFrame([stat], columns=metric_names)
        else:
            df = pd.concat([df, pd.DataFrame([stat], columns=metric_names)], ignore_index=True)

    df.to_csv(outfile, index=False)

class AlbumentationsTransform:
    def __init__(self, is_train=True, size=640):
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=size, width=size, p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=size, width=size, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.40), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        self.image_files = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # 使用 OpenCV 读取图片 (BGR -> RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # YOLO format: class_id, x_center, y_center, width, height
                    cls, x, y, bw, bh = map(float, line.strip().split())

                    # Convert to Pascal VOC format (x_min, y_min, x_max, y_max)
                    x_min = (x - bw / 2) * w
                    y_min = (y - bh / 2) * h
                    x_max = (x + bw / 2) * w
                    y_max = (y + bh / 2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls) + 1)  # Faster R-CNN 背景为0，所以类别从1开始

        if self.transform:
            transformed = self.transform(image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # 转换为 PyTorch Tensors
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        # 如果没有 box，boxes 的 shape 会是 (0,)，需要改为 (0, 4)
        if not boxes:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)

        # 根据 torchvision 要求，计算 area 和 iscrowd
        if boxes:
            area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        else:
            area = torch.tensor([])

        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return image, target


def create_model(backbone_name='resnet50', num_classes=21):
    models = {
        'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'mobilenet320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    }

    model = models[backbone_name](weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model
