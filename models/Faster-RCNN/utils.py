import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
torch.hub.set_dir('./') #修改缓存路径
import torchvision
from torchvision.tv_tensors import BoundingBoxes
from torchvision.models.detection import faster_rcnn
import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class YoloDataset(torch.utils.data.Dataset):
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
            "boxes": BoundingBoxes(boxes, format="xyxy", canvas_size=(h, w)),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w])
        }

        # 根据 torchvision 要求，计算 area 和 iscrowd
        if boxes:
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        else:
            # 如果没有 box，boxes 的 shape 会是 (0,)，需要改为 (0, 4)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target["area"] = torch.tensor([])

        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return image, target

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='best.pth'):
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
            print(f'Val score improved ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ...')
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
    dataset = {"info": {}, "licenses": [], "images": [], "categories": [], "annotations": [] }

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

    # 优化点：使用列表收集所有预测，以便后续进行向量化处理
    scores, labels, boxes, image_ids = [], [], [], []

    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # 收集这个批次的所有预测结果
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            num_preds = len(output["scores"])
            if num_preds == 0:
                continue

            scores.append(output["scores"])
            labels.append(output["labels"])
            boxes.append(output["boxes"])
            # 为该图片的所有预测框重复记录其 image_id
            image_ids.extend([image_id] * num_preds)

    # 如果模型没有任何预测，则提前返回
    if not image_ids:
        print("No predictions were made, skipping COCO evaluation.")
        return None

    # 向量化处理所有收集到的预测结果
    # 将多个Tensor拼接成一个大的Tensor
    scores_tensor = torch.cat(scores, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    boxes_tensor = torch.cat(boxes, dim=0)

    # 对所有 boxes 的坐标进行一次性批量转换 (xyxy -> xywh)
    boxes_tensor[:, 2:] -= boxes_tensor[:, :2]

    # 批量转换为Python list
    scores = scores_tensor.tolist()
    labels = labels_tensor.tolist()
    boxes = boxes_tensor.tolist()

    # 使用一次列表推导式高效地构建最终COCO结果
    coco_results = [{
            "image_id": img_id,
            "category_id": label,
            "bbox": box,
            "score": score,
        }for img_id, label, box, score in zip(image_ids, labels, boxes, scores)
    ]

    # 使用 COCO API 进行评估
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if outfile is not None:
        write_coco_stat(coco_eval.stats, outfile)
        pass

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
                A.pytorch.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=size, width=size, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.40), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)


def create_model(backbone='resnet50', num_classes=21):
    models = {
        'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'mobilenet320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    }

    model = models[backbone](weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model
