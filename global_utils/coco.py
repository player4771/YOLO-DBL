import torch
import joblib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = (
    'coco_stat_names',
    'convert_to_coco_api',
    'COCOEvaluator',
)


coco_stat_names = (
        'mAP', 'AP50', 'AP75',
        'APs', 'APm', 'APl',
        'AR1', 'AR10', 'AR100',
        'ARs', 'ARm', 'ARl'
)

def convert_to_coco_api(dataset):
    """将数据集转换为COCO API格式以进行评估"""
    categories, annotations, images = set(), [], []
    ann_id = 1
    for img_idx in range(len(dataset)):
        targets = dataset.get_targets(img_idx)
        image_id = targets["image_id"].item()
        img_dict = {
            'id': image_id,
            'height': targets['orig_size'][0].item(),
            'width': targets['orig_size'][1].item()
        }
        images.append(img_dict)

        #向量化计算宽和高 (xyxy -> xywh)
        boxes = targets["boxes"] #xyxy
        w = boxes[:, 2] - boxes[:, 0] #这里不可直接box[...]-box[...]，因为会修改原数据
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        boxes = torch.stack([boxes[:, 0], boxes[:, 1], w, h], dim=1) #xywh

        boxes = boxes.tolist()
        areas = areas.tolist()
        labels = targets["labels"].tolist()

        for box,label,area in zip(boxes,labels,areas):
            annotation = {
                'image_id': image_id,
                'bbox': box,
                'category_id': label,
                'id': ann_id,
                'area': area,
                'iscrowd': 0
            }
            annotations.append(annotation)
            categories.add(label)
            ann_id += 1

    dataset = {
        'images': images,
        'categories': [{'id': i} for i in sorted(categories)],
        'annotations': annotations,
        'info': {}
    }

    coco_ds = COCO()
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

class COCOEvaluator:
    def __init__(self, outdir:str|Path=None, coco_gt=None):
        self.outdir = Path(outdir) if outdir else None
        if self.outdir:
            self.outdir.mkdir(parents=True, exist_ok=True)
        self.coco_gt = coco_gt
        self.coco_eval = None
        self.coco_stats = pd.DataFrame()
        self.best_score = 0 #mAP50-95

    @torch.inference_mode()
    def evaluate(self, model, data_loader, min_score:float=0.01):
        device = next(model.parameters()).device
        is_training = model.training
        model.eval()
        if self.coco_gt is None:
            print('Warning: suggest to provide coco_gt as argument')
            self.coco_gt = convert_to_coco_api(data_loader.dataset)

        boxes, labels, scores, image_ids = [], [], [], []
        for images, targets in tqdm(data_loader, desc="Eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                output_scores = output['scores']
                output_boxes = output['boxes']
                output_labels = output['labels']

                keep = output_scores > min_score #注：min_score设置为一个大于0的值可以明显提高速度
                if not keep.any():
                    continue
                output_scores = output_scores[keep]
                output_boxes = output_boxes[keep]
                output_labels = output_labels[keep]

                boxes.append(output_boxes.cpu())
                labels.append(output_labels.cpu())
                scores.append(output_scores.cpu())

                #list*int -> torch.full, 创建一个填充了image_id的tensor，长度等于预测框数
                image_ids.append(torch.full((len(output_scores),), target['image_id'].item(), dtype=torch.int64))


        if is_training:
            model.train()

        image_ids = torch.cat(image_ids).tolist()
        boxes = torch.cat(boxes)
        # 向量化坐标转换 (xyxy -> xywh)
        boxes[:, 2:] -= boxes[:, :2]
        boxes = boxes.tolist()
        scores = torch.cat(scores, dim=0).tolist()
        labels = torch.cat(labels, dim=0).tolist()

        coco_results = [{
            "image_id": image_id,
            "category_id": label,
            "bbox": box,
            "score": score,
        } for image_id, label, box, score in zip(image_ids, labels, boxes, scores)]

        if len(coco_results) == 0:
            print("Warning: coco_results is empty, skipping evaluation.")
            return None

        coco_dt = self.coco_gt.loadRes(coco_results)
        self.coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        if self.outdir: #-> save results (if outdir exists)
            #update stats here
            self.coco_stats = pd.concat([
                    self.coco_stats,
                    pd.DataFrame([self.coco_eval.stats], columns=coco_stat_names)
                ], ignore_index=True
            )
            self.coco_stats.to_csv(self.outdir/'coco_stats.csv', index=False) #update stats file
            if self.coco_eval.stats[0] > self.best_score:
                cocoeval_best = {
                    'stats': self.coco_eval.stats,
                    'eval': self.coco_eval.eval,
                    'params': self.coco_eval.params
                }
                joblib.dump(cocoeval_best, self.outdir/'cocoeval_best.bin')

        return self.coco_eval
