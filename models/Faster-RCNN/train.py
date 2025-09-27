# train.py
import torch
import torch.optim as optim
from tqdm import tqdm
from model import FasterRCNN
from utils import parse_data_yaml, EarlyStopping
from dataset import create_dataloader

# --- Config ---
CONFIG = {
    "data_yaml": "E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
    "project":'./runs/train',
    "img_size": (640, 640),
    "batch_size": 4,  # 根据你的 GPU 显存调整
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":8
}


def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for imgs, bboxes, labels, _ in progress_bar:
        imgs = imgs.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]

        # **【优化2】** 使用 set_to_none=True 加速梯度清零
        optimizer.zero_grad(set_to_none=True)

        # **【优化3】** 使用 AMP 的 autocast 上下文管理器
        with torch.amp.autocast(device_type='cuda'):
            rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss = model(imgs, bboxes, labels)
            loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        if torch.isnan(loss):
            print("Warning: NaN loss detected. Skipping batch.")
            continue

        # **【优化4】** GradScaler 缩放损失并反向传播
        scaler.scale(loss).backward()
        # **【优化5】** GradScaler 更新权重
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for imgs, bboxes, labels, _ in progress_bar:
            imgs = imgs.to(device)
            bboxes = [b.to(device) for b in bboxes]
            labels = [l.to(device) for l in labels]

            # 验证时也使用 autocast，但不需要 scaler
            with torch.cuda.amp.autocast():
                rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss = model(imgs, bboxes, labels)
                loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

            if not torch.isnan(loss):
                total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def main():
    # **【优化6】** 开启 CUDNN 自动调优，加速卷积计算
    if CONFIG['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True

    data_info = parse_data_yaml(CONFIG["data_yaml"])
    n_class = data_info['nc']

    device = torch.device(CONFIG["device"])
    model = FasterRCNN(n_class=n_class).to(device)

    train_loader = create_dataloader(data_info['train'], CONFIG['batch_size'], CONFIG['img_size'],
                                     CONFIG['num_workers'], shuffle=True)
    val_loader = create_dataloader(data_info['val'], CONFIG['batch_size'], CONFIG['img_size'], CONFIG['num_workers'],
                                   shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], fused=True)
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path=CONFIG['project'])

    # **【优化7】** 创建 GradScaler 实例用于混合精度训练
    scaler = torch.amp.GradScaler()

    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training finished.")


if __name__ == '__main__':
    main()