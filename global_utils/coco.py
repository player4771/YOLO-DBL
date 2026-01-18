import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = (
    'coco_stat_names',
    'convert_to_coco_api',
    'COCOEvaluator',
    'get_coco_PRF1',
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

def get_coco_PRF1(cocoeval_file:str|Path):
    data = joblib.load(cocoeval_file)

    # 获取 precision 矩阵
    # 形状: [T*R*K*A*M]
    # T(IoU): 10个 (0=0.50, 1=0.55, ..., 9=0.95)
    # R(Recall): 101个 (0.00, 0.01, ..., 1.00)
    # K(Class): 类别数量
    # A(Area): 4个 (0=all, 1=small, 2=medium, 3=large)
    # M(MaxDets): 3个 (0=1, 1=10, 2=100)
    precision_matrix = data['eval']['precision']

    # 获取 scores 矩阵 (对应每个 P-R 点的置信度阈值)
    # 某些旧版本 pycocotools 可能不包含 scores，做个防错处理
    scores_matrix = data['eval'].get('scores', None)

    # 设置评估标准 (Strict Definition)
    iou_idx = 0  # 严格使用 IoU = 0.50
    area_idx = 0  # 面积 = all
    max_dets_idx = 2  # MaxDets = 100

    # 提取特定维度的数据 -> 形状变为 [Recall(101), Class(K)]
    # 如果某类别没数据，值为 -1
    p_curve = precision_matrix[iou_idx, :, :, area_idx, max_dets_idx]
    s_curve = scores_matrix[iou_idx, :, :, area_idx, max_dets_idx] if scores_matrix is not None else None

    # 生成标准的 Recall 轴 (0 到 1，共 101 个点)
    r_curve = np.linspace(0, 1, 101)  # Shape: (101,)
    # 扩展 Recall 维度以匹配类别数: [101, K]
    r_curve = np.tile(r_curve[:, None], (1, p_curve.shape[1]))

    # 计算 F1 曲线
    # F1 = 2 * P * R / (P + R)
    # 处理 P = -1 的情况 (无数据)，设为 0 以免报错
    valid_mask = p_curve > -1
    p_curve_clean = np.where(valid_mask, p_curve, 0.0)

    # 计算 F1 (避免除以 0)
    denominator = p_curve_clean + r_curve
    f1_curve = np.divide(
        2 * p_curve_clean * r_curve,
        denominator,
        out=np.zeros_like(p_curve_clean),
        where=denominator > 1e-6
    )

    # 寻找最佳点 (Best F1)
    num_classes = p_curve.shape[1]

    results = []
    f1_sum, p_sum, r_sum, valid_classes = 0, 0, 0, 0
    print(f"{'Class ID':<8} | {'Precision':<9} | {'Recall':<9} | {'Best F1':<9} | {'Threshold':<9}")
    for k in range(num_classes):
        # 获取该类别的 F1 曲线
        f1_k = f1_curve[:, k]

        # 找到 F1 最大的索引
        best_idx = np.argmax(f1_k)

        best_f1 = f1_k[best_idx]
        best_p = p_curve_clean[best_idx, k]
        best_r = r_curve[best_idx, k]
        best_s = s_curve[best_idx, k] if s_curve is not None else 0.0

        results.append((k, best_f1, best_p, best_r))

        f1_sum += best_f1
        p_sum += best_p
        r_sum += best_r
        valid_classes += 1

        print(f"{k:<8} | {best_p:.4f}    | {best_r:.4f}    | {best_f1:.4f}    | {best_s:.4f}")

    # 计算平均值 (Macro Average)
    avg_f1 = f1_sum / valid_classes
    avg_p = p_sum / valid_classes
    avg_r = r_sum / valid_classes

    print(f"{'MEAN':<8} | {avg_p:.4f}    | {avg_r:.4f}    | {avg_f1:.4f}    | {'-':<9}")
    print("*MEAN 为各类别 Best F1 点的算术平均值。")


if __name__ == '__main__':
    get_coco_PRF1(r"E:\Projects\PyCharm\Paper2\models\Faster-RCNN\runs\train11\cocoeval_best.bin")