import yaml
from tqdm import tqdm
from pathlib import Path
from torch.backends import cudnn
cudnn.benchmark = True
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW
from torch.amp import GradScaler, autocast

from utils import find_new_dir, EarlyStopping, YoloDataset, AlbumentationsTransform, create_model, evaluate

def collate_fn(batch):
    """
    DataLoader 的 collate_fn，因为每张图的 box 数量不同。
    """
    return tuple(zip(*batch))


def train(**kwargs):
    cfg = {
        'backbone': 'resnet50',
        'data': None,
        'project': './runs',
        'name': 'train',
        'epochs': 20,
        'lr': 1e-3,
        'lf': 1e-2,
        'batch_size': 8,
        'num_workers': 8,
        'weight_decay': 1e-5,
        'patience': 10,
        'device': 'cuda',
        'warmup': None,  # 预热epoch数，None为禁用
        'img_size': 640
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg.update(yaml.load(infile, Loader=yaml.FullLoader))

    num_classes = cfg['nc'] + 1
    output_dir = find_new_dir(Path(cfg['project'], cfg['name']))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    data_path = Path(cfg['data']).parent
    train_img_path = data_path / cfg['train']
    val_img_path = data_path / cfg['val']
    train_label_path = train_img_path.parent / 'labels'
    val_label_path = val_img_path.parent / 'labels'
    results_file = output_dir / 'results.csv'

    dataset_train = YoloDataset(
        img_dir=str(train_img_path),
        label_dir=str(train_label_path),
        transform=AlbumentationsTransform(is_train=True, img_size=cfg['img_size'])
    )
    dataset_val = YoloDataset(
        img_dir=str(val_img_path),
        label_dir=str(val_label_path),
        transform=AlbumentationsTransform(is_train=False, img_size=cfg['img_size'])
    )

    train_loader = DataLoader(
        dataset_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )

    model = create_model(backbone_name=cfg['backbone'], num_classes=num_classes)
    model.to(cfg['device'])

    scaler = GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * cfg['lf'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup'] * len(train_loader) if cfg['warmup'] else 0
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)

    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for (images, targets) in pbar:
            images = [image.to(cfg['device'], non_blocking=True) for image in images]
            targets =[{k: v.to(cfg['device'], non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(device_type=cfg['device']):
                # Faster R-CNN 在训练模式下直接返回 loss dict
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            warmup_scheduler.step()

            pbar.set_postfix(loss=losses.item())

        if not cfg['warmup'] or epoch >= cfg['warmup']:
            scheduler.step()

        coco_eval = evaluate(model, val_loader, cfg['device'], outfile=results_file)
        # 使用 COCO 标准的 mAP (IoU=0.50:0.95) 作为评估指标
        mAP = coco_eval.stats[0] if coco_eval else 0.0

        early_stopper.update(mAP, model)
        if early_stopper.early_stop:
            print(f'Early stopping with mAP {mAP:.6f}')
            break

    print(f"Training finished, Results saved to {results_file}.")


if __name__ == '__main__':
    train(
        backbone='resnet50v2',
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=100,
        patience=10,
        lr=1e-3,
        warmup=1,
        batch_size=8,
        num_workers=4,
        img_size=300,
    )