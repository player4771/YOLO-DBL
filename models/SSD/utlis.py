import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import v2
from torchvision.models import detection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import save, load, inference_mode, float32, stack, full_like, cat

from src import Backbone, SSD300

def create_model(backbone:str, num_classes:int):# -> model
    if backbone == "vgg16":
        model = detection.ssd300_vgg16(weights=detection.SSD300_VGG16_Weights.COCO_V1)
        # 替换分类头以匹配类别数
        model.head.classification_head = detection.ssd.SSDClassificationHead(
            in_channels=[512, 1024, 512, 256, 256, 256],  # For VGG16 backbone in SSD300
            num_anchors=[4, 6, 6, 6, 4, 4],  # Default anchors for SSD300
            num_classes=num_classes,
        )
    elif backbone == "resnet50":
        model = SSD300(backbone=Backbone(), num_classes=num_classes)
        pre_model_dict = load("./nvidia_ssdpyt_amp_200703.pt", map_location='cpu')
        pre_weights_dict = pre_model_dict["model"]
        # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
        del_conf_loc_dict = {}
        for k, v in pre_weights_dict.items():
            split_key = k.split(".")
            if "conf" in split_key:
                continue
            del_conf_loc_dict.update({k: v})
        model.load_state_dict(del_conf_loc_dict, strict=False)
    else:
        raise "Invalid backbone, required 'vgg16' or 'resnet50'."

    return model

class SSDTransform:
    def __init__(self, is_train=True):
        if is_train:
            # 为训练集构建一个强大的数据增强管道
            self.transform = v2.Compose([
                # 随机调整亮度和对比度等
                v2.RandomPhotometricDistort(p=0.8),
                # 随机扩展画布，制造小目标
                v2.RandomZoomOut(),
                # 随机裁剪，同时保证至少有一个目标
                #v2.RandomIoUCrop(),这个有bug
                # 随机水平翻转
                v2.RandomHorizontalFlip(p=0.5),
                # 转换为张量并归一化到 [0, 1]
                v2.ToImage(), #不加会报错
                v2.ToDtype(float32, scale=True),
            ])
        else:
            # 验证集通常只做最基础的转换
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(float32, scale=True),
            ])

    def __call__(self, image, target):
        # v2 的变换可以同时应用于图像和标注
        image, target = self.transform(image, target)
        return image, target

class EarlyStopping:
    """
    当监控的指标停止改善时，提前停止训练。
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): 在停止训练前，等待多少个 epoch 没有改善。
            verbose (bool): 如果为 True，则为每次改善打印一条信息。
            delta (float):  被认为是改善的最小变化量。
            path (str):     保存最佳模型的路径。
            trace_func (function): 用于打印信息的函数。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def update(self, val_metric, model):
        # 我们监控的是 mAP，所以分数越高越好
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta: # 指标没有改善
            self.counter += 1
            self.trace_func(f'EarlyStopping : {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # 指标改善
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """当验证指标改善时，保存模型。"""
        if self.verbose:
            self.trace_func(f'Val metric improved: {self.val_metric_min:.6f} --> {val_metric:.6f}.  Saving model ...')
        save(model.state_dict(), self.path)
        self.val_metric_min = val_metric


def convert_to_coco_api(ds):
    """将数据集转换为COCO API格式以进行评估"""
    coco_ds = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': [], 'info':{}}
    categories = set()
    for img_idx in range(len(ds)):
        _, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {
            'id': image_id,
            'height': targets['orig_size'][0].item(),
            'width': targets['orig_size'][1].item()
        }
        dataset['images'].append(img_dict)

        boxes = targets["boxes"]
        labels = targets["labels"]
        for i in range(boxes.shape[0]):
            box = boxes[i].tolist()
            label = labels[i].item()
            categories.add(label)
            ann = {'image_id': image_id,
                   'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                   'category_id': label,
                   'id': ann_id,
                   'iscrowd': 0
            }
            ann['area'] = ann['bbox'][2] * ann['bbox'][3]
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

@inference_mode()
def evaluate(model, data_loader, device, coco_gt=None, outfile=None):
    """使用 pycocotools 进行评估"""
    model.eval()
    #每次评估都计算一次gt没有必要，可以作为一个固定参数传入
    if coco_gt is None:
        print('Warning: suggest to provide coco_gt as argument')
        coco_gt = convert_to_coco_api(data_loader.dataset)
    coco_dt = []

    for images, targets in tqdm(data_loader, desc="Eval"):
        images = list(img.to(device) for img in images)

        outputs = model(images)

        for target, output in zip(targets, outputs):
            # 获取单个图像的 image_id 和预测结果
            image_id = target['image_id'].item()
            scores = output['scores']

            # 使用向量化操作进行分数过滤
            keep = scores > 0.05
            boxes = output['boxes'][keep]
            labels = output['labels'][keep]
            scores = scores[keep]

            if boxes.numel() == 0:
                continue

            # 使用向量化操作将 boxes 从 xyxy 转换为 xywh
            # boxes[:, 2] 代表所有框的 xmax, boxes[:, 0] 代表 xmin
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = xmax - xmin
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = ymax - ymin

            # 将 image_id 扩展为与检测框数量匹配的张量
            image_ids = full_like(labels, fill_value=image_id)

            # 将当前图片的所有处理结果打包，并转移到 CPU
            # torch.stack 会创建一个新的维度
            processed_output = stack([
                image_ids, labels, scores,
                boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            ], dim=1)

            # cat/stack 之后再 .cpu() 效率更高
            coco_dt.append(processed_output.cpu())

    if not coco_dt:
        print("No predictions made, skipping evaluation.")
        return None

    # 将列表中的所有 tensor 合并为一个
    all_predictions = cat(coco_dt, dim=0).numpy()

    coco_dt_final = []
    for pred in all_predictions:
        coco_dt_final.append({
            'image_id': int(pred[0]),
            'category_id': int(pred[1]),
            'score': pred[2],
            'bbox': [pred[3], pred[4], pred[5], pred[6]],
        })

    coco_dt = coco_gt.loadRes(coco_dt_final)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if outfile is not None:
        write_coco_stat(coco_eval.stats, outfile)

    # 返回 coco_eval.stats[0]，即 AP @ IoU=0.50:0.95
    return coco_eval


def find_new_dir(name:str) -> str:
    num = re.search(r'\d+$', name)
    if num: #结尾有数字：序号+1
        return name[:num.start()]+str(int(num.group(0))+1)
    else: #结尾没数字：添加序号2
        return name+'2'

def write_coco_stat(stat:list, outfile:str|Path) -> None:
    metric_names = [
        'mAP', 'AP50', 'AP75',
        'APs', 'APm', 'APl',
        'ARmax=1', 'ARmax=10', 'ARmax=100',
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
            df = pd.concat([df, pd.DataFrame([stat])], ignore_index=True)

    df.to_csv(outfile, index=False)