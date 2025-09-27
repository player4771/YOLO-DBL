import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import re

from model import FastRCNN
from utils import get_dataloaders, parse_data_cfg, convert_and_save_fp16


def get_project_dir(base_project):
    """为本次训练创建一个新的实验目录，例如 runs/train/exp, runs/train/exp2 ..."""
    if not os.path.exists(base_project):
        return base_project
    parent_dir = os.path.dirname(base_project)
    if not parent_dir:
        parent_dir = '.'
    project_name = os.path.basename(base_project)

    dirs = [d for d in os.listdir(parent_dir) if d.startswith(project_name)]
    matches = [re.match(rf"{project_name}(\d*)", d) for d in dirs]
    nums = [int(m.group(1)) for m in matches if m and m.group(1) and m.group(1).isdigit()]
    n = max(nums) + 1 if nums else 2
    return f"{base_project}{n}"


def evaluate(model, val_loader, device, cfg):
    """在验证集上评估模型"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, rois, target_labels, target_deltas in val_loader:
            images, rois, target_labels, target_deltas = \
                images.to(device), rois.to(device), target_labels.to(device), target_deltas.to(device)

            # 在验证时也使用 autocast，以匹配训练时的数值精度
            with torch.amp.autocast(device_type='cuda', enabled=cfg['amp']):
                scores, bbox_deltas = model(images, rois)
                cls_loss = F.cross_entropy(scores, target_labels)

                pos_mask = (target_labels < (cfg['num_classes'] - 1))
                num_pos = pos_mask.sum()

                if num_pos > 0:
                    # 注意：在获取回归损失的目标时，需要正确索引
                    pred_deltas = bbox_deltas.view(-1, cfg['num_classes'], 4)[
                        torch.arange(len(pos_mask)), target_labels]
                    reg_loss = F.smooth_l1_loss(pred_deltas[pos_mask], target_deltas[pos_mask],
                                                reduction='sum') / num_pos
                else:
                    reg_loss = torch.tensor(0.0, device=device)

            total_val_loss += (cls_loss + reg_loss).item()

    return total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0


def train(**kwargs):
    """主训练函数"""
    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'project': 'runs/train/exp',
        'epochs': 50,
        'lr': 0.001,
        'img_size': 640,
        'batch_size': 8,
        'early_stopping_patience': 10,
        'roi_num_samples': 128,
        'roi_pos_fraction': 0.25,
        'roi_pos_iou_thresh': 0.5,
        'roi_neg_iou_thresh': 0.5,
        'num_workers': 4,
        'amp': True,  # 启用自动混合精度
    }
    cfg.update(kwargs)

    device = torch.device(cfg['device'])
    project_dir = get_project_dir(cfg['project'])
    os.makedirs(project_dir, exist_ok=True)

    # --- 核心修改：提前解析数据配置，以在创建数据集缓存前获取 num_classes ---
    data_info = parse_data_cfg(cfg['data'])
    cfg['num_classes'] = data_info['nc'] + 1

    # 数据加载器现在会利用 utils.py 中的缓存逻辑自动创建或加载缓存
    train_loader, val_loader = get_dataloaders(cfg, data_info)

    model = FastRCNN(num_classes=cfg['num_classes']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # --- 核心修改：初始化混合精度训练的 GradScaler ---
    scaler = torch.amp.GradScaler(enabled=cfg['amp'])

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(project_dir, 'best.pth')

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for i, (images, rois, target_labels, target_deltas) in enumerate(pbar):
            images, rois, target_labels, target_deltas = \
                images.to(device), rois.to(device), target_labels.to(device), target_deltas.to(device)

            # --- 核心修改：使用 autocast 上下文管理器 ---
            with torch.amp.autocast(device_type='cuda', enabled=cfg['amp']):
                scores, bbox_deltas = model(images, rois)
                cls_loss = F.cross_entropy(scores, target_labels)
                pos_mask = (target_labels < (cfg['num_classes'] - 1))
                num_pos = pos_mask.sum()

                if num_pos > 0:
                    pred_deltas = bbox_deltas.view(-1, cfg['num_classes'], 4)[
                        torch.arange(len(pos_mask)), target_labels]
                    reg_loss = F.smooth_l1_loss(pred_deltas[pos_mask], target_deltas[pos_mask],
                                                reduction='sum') / num_pos
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                loss = cls_loss + reg_loss

            optimizer.zero_grad()
            # --- 核心修改：使用 scaler 来缩放损失并反向传播 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(train_loss=f'{total_loss / (i + 1):.4f}')

        val_loss = evaluate(model, val_loader, device, cfg)
        print(
            f"\nEpoch {epoch + 1}/{cfg['epochs']}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{cfg['early_stopping_patience']}")

        if patience_counter >= cfg['early_stopping_patience']:
            print("Early stopping triggered.")
            break
        scheduler.step()

    print(f"Training finished. Best model is at {best_model_path}")

    if os.path.exists(best_model_path):
        # 调用转换函数，自动保存一个_fp16.pth文件
        convert_and_save_fp16(best_model_path, num_classes=cfg['num_classes'])
    else:
        print("No best model was saved. Skipping fp16 conversion.")


if __name__ == '__main__':
    train(
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project='./runs/train',
        batch_size=8,
        lr=1e-3,
        epochs=100,
        early_stopping_patience=10,
        num_workers=8
    )