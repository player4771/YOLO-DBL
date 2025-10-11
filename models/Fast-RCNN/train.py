import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
torch.hub.set_dir('./') #修改缓存路径

from model import FastRCNNFPN as FastRCNN
from utils import convert_and_save_fp16, evaluate, FastRCNNCollator
from global_utils import find_new_dir, EarlyStopping, get_dataloader, YoloDataset, AlbumentationsTransform, this_time


def train(**kwargs):
    cfg = {
        'project': 'runs/train',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'amp': True,
        'epochs': 50,
        'patience': 5,
        'lr': 0.001,
        'img_size': 640,
        'batch_size': 8,
        'num_workers': 4,
        'weight_decay': 1e-5,
        'roi_num_samples': 128,
        'roi_pos_fraction': 0.25,
        'roi_pos_iou_thresh': 0.5,
        'roi_neg_iou_thresh': 0.5,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as f:
        cfg['dataset'] = yaml.load(f, yaml.FullLoader)

    cfg['num_classes'] = dict(cfg['dataset'])['nc'] + 1
    device = torch.device(cfg['device'])
    output_dir = find_new_dir(Path(cfg['project']))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output = Path(output_dir, 'best.pth')

    cfg['start_time'] = this_time()
    with open(output_dir/'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    train_loader = get_dataloader(cfg, YoloDataset, AlbumentationsTransform, True, FastRCNNCollator(cfg))
    val_loader = get_dataloader(cfg, YoloDataset, AlbumentationsTransform, False, FastRCNNCollator(cfg))

    model = FastRCNN(num_classes=cfg['num_classes']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler(device=cfg['device'], enabled=cfg['amp'])
    early_stopper = EarlyStopping(patience=cfg['patience'], path=str(model_output))

    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{cfg['epochs']}")
        for (images, rois, target_labels, target_deltas) in pbar:
            images, rois, target_labels, target_deltas = \
                images.to(device), rois.to(device), target_labels.to(device), target_deltas.to(device)

            with torch.amp.autocast(device_type=cfg['device'], enabled=cfg['amp']):
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(train_loss=f'{loss.item():.4f}')

        val_loss = evaluate(model, val_loader, device, cfg)
        #print(f"\nEval {epoch + 1}/{cfg['epochs']}, Val Loss: {val_loss:.4f}")

        early_stopper.update(-val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        scheduler.step()

    print(f"Training finished. Best model saved to {model_output}")

    if model_output.exists():
        # 调用转换函数，自动保存一个_fp16.pth文件
        convert_and_save_fp16(str(model_output), num_classes=cfg['num_classes'])
    else:
        print("No best model was saved. Skipping fp16 conversion.")


if __name__ == '__main__':
    train(
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project='./runs/train',
        lr=1e-4,
        epochs=100,
        patience=10,
        batch_size=8,
        num_workers=8
    )