import yaml
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark=True
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from backbone import ResNetBackbone
from global_utils import (EarlyStopping, YoloDataset, AlbumentationsTransform,
                          find_new_dir, evaluate, convert_to_coco_api, WindowsSleepAvoider)

def create_model(backbone:str='vgg16', num_classes:int=4): # -> model
    if backbone == "vgg16":
        model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT')
        # 替换分类头以匹配类别数
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=[512, 1024, 512, 256, 256, 256],  # For VGG16 backbone in SSD300
            num_anchors=[4, 6, 6, 6, 4, 4],  # Default anchors for SSD300
            num_classes=num_classes,
        )
    elif backbone == "resnet50":
        resnet_backbone = ResNetBackbone()

        # 定义每个特征图的输出通道数
        # 对应 ResNetBackbone 的 layer2, layer3, extra1, extra2, extra3, extra4
        out_channels = [512, 1024, 512, 256, 256, 256]

        # 创建 Anchor Generator
        # SSD300 使用 6 个特征图，每个特征图的 anchor 数量
        num_anchors = [4, 6, 6, 6, 4, 4]
        anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator(
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )

        # 创建 SSD 的检测头
        head = torchvision.models.detection.ssd.SSDHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

        # 组装成完整的 SSD 模型
        model = torchvision.models.detection.ssd.SSD(
            backbone=resnet_backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            head=head,
            size=(300,300)
        )
    else:
        raise "Invalid backbone, required 'vgg16' or 'resnet50'."

    return model


def collate_fn(batch):
    """
    DataLoader 的 collate_fn，因为每张图的 box 数量不同。
    """
    return tuple(zip(*batch))


def train(**kwargs):
    cfg = { #default args
        'backbone':'vgg16',
        'data':None, #不可缺省
        'dataset': {}, #避免类型检查警告
        'project':'./runs',
        'name':'train',
        'epochs':20,
        'lr':1e-3,
        'lf':1e-2,
        'batch_size':8,
        'num_workers':8,
        'weight_decay':1e-5,
        'patience':5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'warmup_epochs': 0,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg['dataset'] = yaml.load(infile, Loader=yaml.FullLoader)

    num_classes = cfg['dataset']['nc'] + 1 # +1 是因为 SSD 需要一个背景类
    device = torch.device(cfg['device'])
    output_dir = find_new_dir(Path(cfg['project'], cfg['name']))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg['start_time'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    with open(output_dir/'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    data_root = Path(cfg['data']).parent
    train_img_dir = data_root / cfg['dataset']['train']
    val_img_dir = data_root / cfg['dataset']['val']
    train_label_dir = train_img_dir.parent / 'labels'
    val_label_dir = val_img_dir.parent / 'labels'
    results_file = output_dir / 'results.csv'

    # 创建 Dataset
    dataset_train = YoloDataset(
        img_dir=train_img_dir, label_dir=train_label_dir,
        transform=AlbumentationsTransform(is_train=True, size=300)
    )
    dataset_val = YoloDataset(
        img_dir=val_img_dir, label_dir=val_label_dir,
        transform=AlbumentationsTransform(is_train=False, size=300)
    )

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        pin_memory=True, shuffle=True, persistent_workers=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        pin_memory=True, shuffle=False, persistent_workers=True,
        collate_fn=collate_fn
    )

    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)
    model.to(device)

    scaler = torch.amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * cfg['lf'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup_epochs']*len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)
    coco_gt = convert_to_coco_api(val_loader.dataset)

    fucker = WindowsSleepAvoider(delay=600, distance=200)
    fucker.start()

    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for (images, targets) in pbar:
            images = torch.stack(images, dim=0).to(device=device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=cfg['device']):
                results = model(images, targets)
                losses = torch.Tensor(sum(loss for loss in results.values()))

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch < cfg['warmup_epochs']:
                warmup_scheduler.step()

            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=losses.item())

        if epoch >= cfg['warmup_epochs']:
            scheduler.step()

        coco_eval = evaluate(model, val_loader, device, coco_gt=coco_gt, outfile=results_file)
        mAP = coco_eval.stats[0] if coco_eval else 0.0

        early_stopper.update(mAP, model)
        if early_stopper.early_stop:
            print(f'Early stopping with mAP {mAP:.6f}')
            break

    print(f"Training finished, Results saved to {results_file}.")
    fucker.stop()


if __name__ == '__main__':
    train(
        backbone='resnet50',
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=100,
        patience=10,
        lr=1e-3,
        warmup_epochs=1,
        batch_size=8,
        num_workers=4
    )

