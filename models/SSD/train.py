import yaml
from tqdm import tqdm
from pathlib import Path

from torch import stack
from torch.hub import set_dir
set_dir('./')
from torch.backends import cudnn
cudnn.benchmark=True
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW
from torch.amp import GradScaler, autocast

from dataset import YoloDataset
from utlis import create_model, SSDTransform, EarlyStopping, evaluate, find_new_dir, convert_to_coco_api


def collate_fn(batch):
    """
    DataLoader 的 collate_fn，因为每张图的 box 数量不同。
    """
    return tuple(zip(*batch))


def train(**kwargs):
    cfg={ #default args
        'backbone':'vgg16',
        'data':None, #不可缺省
        'project':'./runs',
        'name':'train',
        'epochs':20,
        'lr':1e-3,
        'lf':1e-2,
        'batch_size':8,
        'num_workers':8,
        'weight_decay':1e-5,
        'patience':5,
        'device':'cuda',
        'warmup':None #预热epoch数，None为禁用
    }

    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg.update(yaml.load(infile, Loader=yaml.FullLoader))

    num_classes = cfg['nc'] + 1 # +1 是因为 SSD 需要一个背景类
    output_dir = Path(cfg['project'], cfg['name'])
    while output_dir.exists(): #若路径已存在，则加序号(或序号增加)，寻找新文件夹
        cfg['name'] = find_new_dir(cfg['name'])
        output_dir = Path(cfg['project'], cfg['name'])

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir/'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data_path = Path(cfg['data']).parent
    train_img_path = data_path / cfg['train']
    val_img_path = data_path / cfg['val']
    train_label_path = train_img_path.parent / 'labels'
    val_label_path = val_img_path.parent / 'labels'
    results_file = output_dir/'results.csv'

    # 创建 Dataset
    dataset_train = YoloDataset(
        img_dir=str(train_img_path),
        label_dir=str(train_label_path),
        transform=SSDTransform(is_train=True)
    )
    dataset_val = YoloDataset(
        img_dir=str(val_img_path),
        label_dir=str(val_label_path),
        transform=SSDTransform(is_train=False)
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )

    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)
    model.to(cfg['device'])

    scaler = GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    optimizer = AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * cfg['lf'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup']*len(train_loader) if cfg['warmup'] else 0
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)
    coco_gt = convert_to_coco_api(val_loader.dataset)


    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for (images, targets) in pbar:
            #images = list(image.to(cfg['device'], non_blocking=True) for image in images)
            images = stack(images, dim=0).to(device=cfg['device'], non_blocking=True)
            targets = [{k: v.to(cfg['device'], non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(device_type=cfg['device']):
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

        coco_eval = evaluate(model, val_loader, cfg['device'], coco_gt=coco_gt, outfile=results_file)
        mAP = coco_eval.stats[0] if coco_eval else 0.0

        early_stopper.update(mAP, model)
        if early_stopper.early_stop:
            print(f'Early stopping with mAP {mAP:.6f}')
            break

    print(f"Training finished, Results saved to {results_file}.")


if __name__ == '__main__':
    train(
        backbone='vgg16',
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=100,
        patience=10,
        lr=1e-3,
        warmup=1,
        batch_size=16,
        num_workers=8
    )