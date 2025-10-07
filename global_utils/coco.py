import torch
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .tools import write_coco_stat

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

        boxes, labels = targets["boxes"], targets["labels"]
        # Convert from xyxy to xywh
        boxes[:, 2:] -= boxes[:, :2]

        for i in range(boxes.shape[0]):
            ann = {
                'image_id': image_id,
                'bbox': boxes[i].tolist(),
                'category_id': labels[i].item(),
                'id': ann_id,
                'iscrowd': 0
            }
            categories.add(labels[i].item())
            ann['area'] = ann['bbox'][2] * ann['bbox'][3]
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

@torch.inference_mode()
def evaluate(model, data_loader, device='cpu',
             coco_gt=None, outfile:str|Path=None, min_score:float=None):
    model.eval()
    if coco_gt is None:
        print('Warning: suggest to provide coco_gt as argument')
        coco_gt = convert_to_coco_api(data_loader.dataset)

    boxes, labels, scores, image_ids = [], [], [], []

    for images, targets in tqdm(data_loader, desc="Eval"):
        images = [img.to(device) for img in images]

        outputs = model(images) #on GPU

        for target, output in zip(targets, outputs):
            output_scores = output['scores']

            if min_score:
                # 在GPU上提前过滤，然后只将有用的数据移动到CPU
                keep = output_scores > min_score
                if not keep.any():
                    continue
                boxes.append(output['boxes'][keep].cpu())
                labels.append(output['labels'][keep].cpu())
                scores.append(output_scores[keep].cpu())
                image_id = target['image_id'].item()
                image_ids.extend([image_id] * keep.sum().item())
            else:
                boxes.append(output['boxes'].cpu())
                labels.append(output['labels'].cpu())
                scores.append(output_scores.cpu())
                image_id = target['image_id'].item()
                num_preds = len(output['scores'])
                image_ids.extend([image_id] * num_preds)

    if not image_ids:
        print("No valid predictions after filtering, skipping evaluation.")
        return None

    boxes = torch.cat(boxes, dim=0)
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

    # --- 后续评估代码不变 ---
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if outfile:
        write_coco_stat(coco_eval.stats, outfile)

    return coco_eval