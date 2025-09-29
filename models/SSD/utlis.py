import re
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import save, inference_mode, float32
from torchvision.transforms import v2

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
            self.trace_func(f'val_metric improved: {self.val_metric_min:.6f} --> {val_metric:.6f}.  Saving model ...')
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
def evaluate(model, data_loader, device):
    """使用 pycocotools 进行评估"""
    model.eval()
    coco_gt = convert_to_coco_api(data_loader.dataset)
    coco_dt = []

    pbar = tqdm(data_loader, desc="Eval")
    for images, targets in pbar:
        images = list(img.to(device) for img in images)

        outputs = model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        for target, output in zip(targets, outputs):
            image_id = target['image_id'].item()
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > 0.05:  # score a threshold
                    coco_dt.append({
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': [box[0].item(), box[1].item(),
                                 (box[2] - box[0]).item(), (box[3] - box[1]).item()], #xywh
                        'score': score.item(),
                    })

    if not coco_dt:
        print("No predictions made, skipping evaluation.")
        return None

    coco_dt = coco_gt.loadRes(coco_dt)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回 coco_eval.stats[0]，即 AP @ IoU=0.50:0.95
    return coco_eval


def find_new_dir(name:str) -> str:
    num = re.search(r'\d+$', name)
    if num: #结尾有数字：序号+1
        return name[:num.start()]+str(int(num.group(0))+1)
    else: #结尾没数字：添加序号2
        return name+'2'