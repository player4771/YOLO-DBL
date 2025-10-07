import os
import yaml
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from global_utils import YoloDataset, EarlyStopping, find_new_dir, AlbumentationsTransform, evaluate

def create_model(backbone='resnet50', num_classes=21):
    models = {
        'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'mobilenet320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    }

    model = models[backbone](weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    """DataLoader 的 collate_fn，因为每张图的 box 数量不同。"""
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
        'patience': 5,
        'delta': 1e-3,
        'device': 'cuda',
        'warmup': None,  # 预热epoch数，None为禁用
        'img_size': 640,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg.update(yaml.load(infile, Loader=yaml.FullLoader))

    num_classes = cfg['nc'] + 1
    device = torch.device(cfg['device'])
    output_dir = find_new_dir(Path(cfg['project'], cfg['name']))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir/'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    data_path = Path(cfg['data']).parent
    train_img_path = data_path / cfg['train']
    val_img_path = data_path / cfg['val']
    train_label_path = train_img_path.parent / 'labels'
    val_label_path = val_img_path.parent / 'labels'
    results_file = output_dir / 'results.csv'

    dataset_train = YoloDataset(
        img_dir=str(train_img_path), label_dir=str(train_label_path),
        transform=AlbumentationsTransform(is_train=True, size=cfg['img_size'])
    )
    dataset_val = YoloDataset(
        img_dir=str(val_img_path), label_dir=str(val_label_path),
        transform=AlbumentationsTransform(is_train=False, size=cfg['img_size'])
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )

    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)
    model.to(device)

    scaler = torch.amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * cfg['lf'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, delta=cfg['delta'], path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup'] * len(train_loader) if cfg['warmup'] else 0
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)

    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for (images, targets) in pbar:
            images = [image.to(device, non_blocking=True) for image in images]
            targets= [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=cfg['device'], enabled=torch.cuda.is_available()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            warmup_scheduler.step()

            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=losses.item())

        if not cfg['warmup'] or epoch >= cfg['warmup']:
            scheduler.step()

        coco_eval = evaluate(model, val_loader, device, outfile=results_file)
        mAP = coco_eval.stats[0] if coco_eval else 0.0

        early_stopper.update(mAP, model)
        if early_stopper.early_stop:
            print(f'Early stopping with mAP {mAP:.6f}')
            break

    torch.save(model.state_dict(), output_dir/'last.pth')
    print(f"Training finished, Results saved to {results_file}.")


if __name__ == '__main__':
    train(
        backbone='resnet50v2',
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=100,
        patience=5,
        lr=1e-3,
        warmup=1,
        batch_size=8,
        num_workers=4,
        img_size=300, #不加这个APs和ARs会变为0，原因未知
    )