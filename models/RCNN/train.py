import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
torch.hub.set_dir('./') #修改缓存路径

from model import RCNN
from utils import selective_search, PreprocessedRCNNDataset

from global_utils import EarlyStopping, AlbumentationsTransform, find_new_dir, time_now_str

def create_training_samples_vectorized(proposals, gt_boxes, gt_labels):
    if len(gt_boxes) == 0:
        return np.array([]), proposals, np.array([]), np.array([])
    proposals_exp, gt_boxes_exp = np.expand_dims(proposals, axis=1), np.expand_dims(gt_boxes, axis=0)
    xA, yA = np.maximum(proposals_exp[:, :, 0], gt_boxes_exp[:, :, 0]), np.maximum(proposals_exp[:, :, 1], gt_boxes_exp[:, :, 1])
    xB, yB = np.minimum(proposals_exp[:, :, 2], gt_boxes_exp[:, :, 2]), np.minimum(proposals_exp[:, :, 3], gt_boxes_exp[:, :, 3])
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

class RCNNCollator:
    """在数据加载时动态生成并采样RoI。"""
    def __init__(self, transform, num_samples=128, pos_fraction=0.25):
        self.transform = transform
        self.num_samples = num_samples
        self.pos_samples = int(num_samples * pos_fraction)

    def __call__(self, batch):
        all_rois, all_labels, all_targets = [], [], []
        for image, target in batch:
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy() - 1
            proposals = np.array([[x, y, x + w, y + h] for x, y, w, h in selective_search(image)[:2000]])
            pos_rois, neg_rois, pos_labels, pos_reg_targets = create_training_samples_vectorized(proposals, gt_boxes, gt_labels)

            num_pos = min(len(pos_rois), self.pos_samples)
            num_neg = min(len(neg_rois), self.num_samples - num_pos)

            final_rois, final_labels, final_reg_targets = [], [], []
            if num_pos > 0:
                pos_indices = np.random.choice(len(pos_rois), size=num_pos, replace=False)
                final_rois.extend(pos_rois[pos_indices])
                final_labels.extend(pos_labels[pos_indices])
                final_reg_targets.extend(pos_reg_targets[pos_indices])
            if num_neg > 0:
                neg_indices = np.random.choice(len(neg_rois), size=num_neg, replace=False)
                final_rois.extend(neg_rois[neg_indices])
                final_labels.extend([0] * num_neg)

            for box in final_rois:
                x1, y1, x2, y2 = map(int, box)
                roi_img = image[y1:y2, x1:x2]
                if roi_img.shape[0] > 0 and roi_img.shape[1] > 0:
                    transformed_roi = self.transform(image=roi_img, bboxes=[], class_labels=[])['image']
                    all_rois.append(transformed_roi)

            all_labels.extend(final_labels)
            all_targets.extend(final_reg_targets)

        if not all_rois:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        return torch.stack(all_rois), torch.LongTensor(np.array(all_labels)), torch.FloatTensor(np.array(all_targets))

@torch.inference_mode()
def validate(model, dataloader, device, cls_criterion, reg_criterion, num_classes):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Validation]", leave=False)
        for rois, labels, targets in pbar:
            if rois.nelement() == 0: continue
            rois, labels, targets = rois.to(device), labels.to(device), targets.to(device)
            class_scores, bbox_deltas = model(rois)
            cls_loss = cls_criterion(class_scores, labels)
            pos_indices = torch.where(labels > 0)[0]
            reg_loss = torch.tensor(0.0, device=device)
            if len(pos_indices) > 0:
                pos_labels, pos_targets = labels[pos_indices] - 1, targets[:len(pos_indices)]
                pred_deltas = bbox_deltas.view(-1, num_classes, 4)[pos_indices, pos_labels]
                reg_loss = reg_criterion(pred_deltas, pos_targets)
            loss = cls_loss + reg_loss
            total_val_loss += loss.item()
            pbar.set_postfix(val_loss=f'{loss.item():.4f}')
    return total_val_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def collate_fn(batch):
    """简单的collate_fn，用于堆叠来自PreprocessedRCNNDataset的样本。"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    rois_tensors, labels_tensors, targets_tensors = zip(*batch)
    return torch.cat(rois_tensors, dim=0), torch.cat(labels_tensors, dim=0), torch.cat(targets_tensors, dim=0)


def train(**kwargs):
    cfg = {
        'project': './runs/train',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'amp': True,
        'epochs': 50,
        'patience': 10,
        'lr': 1e-4,
        'batch_size': 2, #多了会爆显存
        'num_workers': 4,
        'weight_decay': 1e-5,
        'unfreeze_layers': 2
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as f:
        cfg['dataset'] = yaml.safe_load(f)
    cfg['num_classes'] = cfg['dataset']['nc']

    device = torch.device(cfg['device'])
    output_dir = find_new_dir(Path(cfg['project']))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output = output_dir / 'best.pth'

    cfg['start_time'] = time_now_str()
    with open(output_dir / 'args.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    yaml_dir = Path(cfg['data']).parent
    train_dir = yaml_dir / 'preprocessed' / 'train'
    val_dir = yaml_dir / 'preprocessed' / 'val'

    # 检查预处理数据是否存在
    if not train_dir.exists() or not val_dir.exists():
        print("Preprocessed data not found.")
        print("Please run preprocess.py first.")
        return

    train_dataset = PreprocessedRCNNDataset(train_dir, transform=AlbumentationsTransform(is_train=True, size=224))
    val_dataset = PreprocessedRCNNDataset(val_dir, transform=AlbumentationsTransform(is_train=False, size=224))

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)

    model = RCNN(num_classes=cfg['num_classes'], unfreeze_layers=cfg['unfreeze_layers']).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    cls_criterion, reg_criterion = nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler(enabled=cfg['amp'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, outfile=str(model_output))

    for epoch in range(cfg['epochs']):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{cfg['epochs']}")
        for rois, labels, targets in pbar:
            if rois.nelement() == 0: continue
            rois, labels, targets = rois.to(device), labels.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=cfg['amp']):
                class_scores, bbox_deltas = model(rois)
                cls_loss = cls_criterion(class_scores, labels)
                pos_indices = torch.where(labels > 0)[0]
                reg_loss = torch.tensor(0.0, device=device)
                if len(pos_indices) > 0:
                    pos_labels, pos_targets = labels[pos_indices] - 1, targets[:len(pos_indices)]
                    pred_deltas = bbox_deltas.view(-1, cfg['num_classes'], 4)[pos_indices, pos_labels]
                    reg_loss = reg_criterion(pred_deltas, pos_targets)
                loss = cls_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = validate(model, val_loader, device, cls_criterion, reg_criterion, cfg['num_classes'])
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{cfg['epochs']}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopper.update(-avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered with loss{avg_val_loss}.")
            break

    print(f"Training Finished, Best model saved to {model_output}")

if __name__ == '__main__':
    train(
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        epochs=100,
    )