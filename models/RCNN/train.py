# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import RCNN
from utils import PreprocessedRCNNDataset, parse_yaml_config, selective_search, EarlyStopping


# --- 预处理相关函数 ---
def create_training_samples_vectorized(proposals, gt_boxes, gt_labels):
    if len(gt_boxes) == 0: return np.array([]), proposals, np.array([]), np.array([])
    proposals_exp, gt_boxes_exp = np.expand_dims(proposals, axis=1), np.expand_dims(gt_boxes, axis=0)
    xA, yA = np.maximum(proposals_exp[:, :, 0], gt_boxes_exp[:, :, 0]), np.maximum(proposals_exp[:, :, 1],
                                                                                   gt_boxes_exp[:, :, 1])
    xB, yB = np.minimum(proposals_exp[:, :, 2], gt_boxes_exp[:, :, 2]), np.minimum(proposals_exp[:, :, 3],
                                                                                   gt_boxes_exp[:, :, 3])
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    prop_areas, gt_areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1]), (
                gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = prop_areas[:, np.newaxis] + gt_areas[np.newaxis, :] - inter_area
    iou_matrix = inter_area / (union_area + 1e-6)
    max_iou_per_proposal, best_gt_idx_per_proposal = np.max(iou_matrix, axis=1), np.argmax(iou_matrix, axis=1)
    IOU_THRESHOLD_POS, IOU_THRESHOLD_NEG = 0.5, 0.1
    positive_indices, negative_indices = np.where(max_iou_per_proposal >= IOU_THRESHOLD_POS)[0], \
    np.where(max_iou_per_proposal < IOU_THRESHOLD_NEG)[0]
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
        t_w, t_h = np.log(g_w / (p_w + 1e-6) + 1e-6), np.log(g_h / (p_h + 1e-6) + 1e-6)
        pos_reg_targets = np.vstack((t_x, t_y, t_w, t_h)).T
    else:
        pos_labels, pos_reg_targets = np.array([]), np.array([])
    return positive_rois, negative_rois, pos_labels, pos_reg_targets


def worker_process(args):
    image_path, label_dir, output_dir = map(str, args)
    image_file = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None: return
    annotation_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    gt_boxes, gt_labels = [], []
    if os.path.exists(annotation_path):
        h, w, _ = image.shape
        with open(annotation_path, 'r') as f:
            for line in f:
                class_id, x_c, y_c, width, height = map(float, line.split())
                x1, y1, x2, y2 = (x_c - width / 2) * w, (y_c - height / 2) * h, (x_c + width / 2) * w, (
                            y_c + height / 2) * h
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(int(class_id))
    proposals = np.array([[x, y, x + w, y + h] for x, y, w, h in selective_search(image)[:2000]])
    pos_rois, neg_rois, pos_labels, pos_reg_targets = create_training_samples_vectorized(proposals, np.array(gt_boxes),
                                                                                         np.array(gt_labels))
    save_data = {'image_path': image_path, 'positive_rois': pos_rois, 'negative_rois': neg_rois,
                 'positive_labels': pos_labels, 'regression_targets': pos_reg_targets}
    torch.save(save_data, os.path.join(output_dir, image_file.replace('.jpg', '.pt').replace('.png', '.pt')))


def preprocess_data_if_needed(yaml_path):
    print("--- Checking for preprocessed data ---")
    yaml_config = parse_yaml_config(yaml_path)
    if not yaml_config: raise ValueError("YAML config could not be parsed.")
    yaml_dir = os.path.dirname(yaml_path)
    data_missing = any(not os.path.exists(os.path.join(yaml_dir, 'preprocessed', split)) or len(
        os.listdir(os.path.join(yaml_dir, 'preprocessed', split))) < len(
        os.listdir(os.path.join(yaml_dir, str(yaml_config[split])))) for split in ['train', 'val'])
    if not data_missing:
        print("Preprocessed data found. Skipping preprocessing.")
        return
    print("Preprocessed data not found or incomplete. Starting parallel preprocessing...")
    num_processes = multiprocessing.cpu_count()//2
    print(f"Using {num_processes} processes for preprocessing.")
    for split in ['train', 'val']:
        print(f"--- Preprocessing '{split}' split ---")
        img_path = str(yaml_config[split])
        img_dir = os.path.join(yaml_dir, img_path)
        label_dir = os.path.join(yaml_dir, img_path.replace('images', 'labels'))
        output_dir = os.path.join(yaml_dir, 'preprocessed', split)
        os.makedirs(output_dir, exist_ok=True)
        image_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith(('.jpg', '.png'))]
        task_args = [(path, label_dir, output_dir) for path in image_paths]
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap_unordered(worker_process, task_args), total=len(task_args), desc=f"Processing {split}"))
    print("--- Preprocessing finished! ---")


# --- 训练核心函数 ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    rois_tensors, labels_tensors, targets_tensors = zip(*batch)
    return torch.cat(rois_tensors, dim=0), torch.cat(labels_tensors, dim=0), torch.cat(targets_tensors, dim=0)


def validate(model, dataloader, device, cls_criterion, reg_criterion, num_classes):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Validation]", leave=False)
        for all_rois, all_labels, all_targets in pbar:
            if all_rois.nelement() == 0: continue
            all_rois, all_labels, all_targets = all_rois.to(device), all_labels.to(device), all_targets.to(device)
            class_scores, bbox_deltas = model(all_rois)
            cls_loss = cls_criterion(class_scores, all_labels)
            pos_indices = torch.where(all_labels > 0)[0]
            reg_loss = torch.tensor(0.0, device=device)
            if len(pos_indices) > 0:
                pos_labels = all_labels[pos_indices] - 1
                pos_targets = all_targets[:len(pos_indices)]
                pred_deltas = bbox_deltas.view(-1, num_classes, 4)[pos_indices, pos_labels]
                reg_loss = reg_criterion(pred_deltas, pos_targets)
            loss = cls_loss + reg_loss
            total_val_loss += loss.item()
            pbar.set_postfix(val_loss=f'{loss.item():.4f}')
    return total_val_loss / len(dataloader) if len(dataloader) > 0 else 0


def train(**kwargs):
    yaml_path = kwargs['yaml_path']
    preprocess_data_if_needed(yaml_path)
    yaml_config = parse_yaml_config(yaml_path)
    if not yaml_config: sys.exit("YAML config could not be parsed.")
    num_classes, yaml_dir = int(yaml_config['nc']), os.path.dirname(yaml_path)
    train_dir, val_dir = os.path.join(yaml_dir, 'preprocessed', 'train'), os.path.join(yaml_dir, 'preprocessed', 'val')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    transform_val = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.22)),
        ToTensorV2(),
    ])

    train_dataset = PreprocessedRCNNDataset(train_dir, transform=transform_train)
    val_dataset = PreprocessedRCNNDataset(val_dir, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True,
                              num_workers=kwargs['num_workers'], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=kwargs['batch_size'], shuffle=False,
                            num_workers=kwargs['num_workers'], pin_memory=True, collate_fn=collate_fn)

    model = RCNN(num_classes=num_classes, unfreeze_layers=kwargs['unfreeze_layers']).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['learning_rate'],
                            weight_decay=kwargs['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs['epochs'], eta_min=1e-6)
    cls_criterion, reg_criterion = nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    early_stopper = EarlyStopping(
        patience=kwargs.get('early_stopping_patience', 7),
        verbose=True,
        path=kwargs['model_save_path']
    )

    print("--- Training Started ---")
    for epoch in range(kwargs['epochs']):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{kwargs['epochs']}")
        for all_rois, all_labels, all_targets in pbar:
            if all_rois.nelement() == 0: continue
            all_rois, all_labels, all_targets = all_rois.to(device), all_labels.to(device), all_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                class_scores, bbox_deltas = model(all_rois)
                cls_loss = cls_criterion(class_scores, all_labels)
                pos_indices = torch.where(all_labels > 0)[0]
                reg_loss = torch.tensor(0.0, device=device)
                if len(pos_indices) > 0:
                    pos_labels = all_labels[pos_indices] - 1
                    pos_targets = all_targets[:len(pos_indices)]
                    pred_deltas = bbox_deltas.view(-1, num_classes, 4)[pos_indices, pos_labels]
                    reg_loss = reg_criterion(pred_deltas, pos_targets)
                loss = cls_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = validate(model, val_loader, device, cls_criterion, reg_criterion, num_classes)
        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{kwargs['epochs']}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"--- Training Finished --- \nBest model saved to {kwargs['model_save_path']}")


if __name__ == '__main__':
    train_config = {
        "yaml_path": "E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        "epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "model_save_path": "./runs/train4/best.pth",
        "num_workers": 8,
        "unfreeze_layers": 2,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10
    }
    assert not os.path.exists(train_config['model_save_path']), "File already exists."
    train(**train_config)