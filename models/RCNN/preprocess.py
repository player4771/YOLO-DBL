import yaml
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path

from utils import selective_search
from global_utils import YOLODataset


def create_training_samples_vectorized(proposals, gt_boxes, gt_labels):
    if len(gt_boxes) == 0:
        return np.array([]), proposals, np.array([]), np.array([])
    proposals_exp, gt_boxes_exp = np.expand_dims(proposals, axis=1), np.expand_dims(gt_boxes, axis=0)
    xA, yA = np.maximum(proposals_exp[:, :, 0], gt_boxes_exp[:, :, 0]), np.maximum(proposals_exp[:, :, 1],
                                                                                   gt_boxes_exp[:, :, 1])
    xB, yB = np.minimum(proposals_exp[:, :, 2], gt_boxes_exp[:, :, 2]), np.minimum(proposals_exp[:, :, 3],
                                                                                   gt_boxes_exp[:, :, 3])
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    prop_areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = prop_areas[:, np.newaxis] + gt_areas[np.newaxis, :] - inter_area
    iou_matrix = inter_area / (union_area + 1e-6)
    max_iou_per_proposal, best_gt_idx_per_proposal = np.max(iou_matrix, axis=1), np.argmax(iou_matrix, axis=1)

    positive_indices = np.where(max_iou_per_proposal >= 0.5)[0]
    negative_indices = np.where(max_iou_per_proposal < 0.1)[0]
    positive_rois, negative_rois = proposals[positive_indices], proposals[negative_indices]

    if len(positive_indices) > 0:
        pos_labels = gt_labels[best_gt_idx_per_proposal[positive_indices]] + 1
        matched_gt_boxes = gt_boxes[best_gt_idx_per_proposal[positive_indices]]
        p_boxes, g_boxes = positive_rois, matched_gt_boxes
        p_w, p_h = p_boxes[:, 2] - p_boxes[:, 0], p_boxes[:, 3] - p_boxes[:, 1]
        p_x, p_y = p_boxes[:, 0] + p_w / 2, p_boxes[:, 1] + p_h / 2
        g_w, g_h = g_boxes[:, 2] - g_boxes[:, 0], g_boxes[:, 3] - g_boxes[:, 1]
        g_x, g_y = g_boxes[:, 0] + g_w / 2, g_boxes[:, 1] + g_h / 2
        t_x, t_y = (g_x - p_x) / p_w, (g_y - p_y) / p_h
        t_w, t_h = np.log(g_w / (p_w + 1e-6)), np.log(g_h / (p_h + 1e-6))
        pos_reg_targets = np.vstack((t_x, t_y, t_w, t_h)).T
    else:
        pos_labels, pos_reg_targets = np.array([]), np.array([])
    return positive_rois, negative_rois, pos_labels, pos_reg_targets


def worker(args):
    image, target, output_dir, image_path = args
    gt_labels = target['labels'].numpy() - 1
    gt_boxes = target['boxes'].numpy()
    proposals = np.array([[x, y, x + w, y + h] for x, y, w, h in selective_search(image)[:2000]])
    pos_rois, neg_rois, pos_labels, pos_reg_targets = create_training_samples_vectorized(proposals, gt_boxes, gt_labels)
    save_data = {'image_path': str(image_path), 'positive_rois': pos_rois, 'negative_rois': neg_rois,
                 'positive_labels': pos_labels, 'regression_targets': pos_reg_targets}
    output_file = output_dir / image_path.with_suffix('.pt').name
    torch.save(save_data, output_file)


def main(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    print("--- Starting data preprocessing ---")
    num_processes = multiprocessing.cpu_count() // 2  # 使用一半的CPU核心
    yaml_dir = Path(data_yaml_path).parent
    for split in ['train', 'val']:
        img_dir = yaml_dir / data_cfg[split]
        label_dir = img_dir.parent / 'labels'
        output_dir = yaml_dir / 'preprocessed' / split
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = YOLODataset(img_dir=img_dir, label_dir=label_dir)
        task_args = [(dataset[i][0], dataset[i][1], output_dir, dataset.img_files[i]) for i in range(len(dataset))]
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(worker, task_args), total=len(task_args), desc=f"Processing {split}"))
    print("--- Preprocessing finished ---")


if __name__ == '__main__':
    main(data_yaml_path="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml")