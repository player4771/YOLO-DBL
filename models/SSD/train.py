import yaml
from tqdm import tqdm
from pathlib import Path

from torch.hub import set_dir
set_dir('./')
from torch.backends import cudnn
cudnn.benchmark=True
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torch.amp import GradScaler, autocast
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from dataset import YoloDataset
from utlis import SSDTransform, EarlyStopping, evaluate, find_new_dir


def collate_fn(batch):
    """
    DataLoader 的 collate_fn，因为每张图的 box 数量不同。
    """
    return tuple(zip(*batch))


def train(**kwargs):
    cfg={
        'data':None, #不可缺省
        'project':'./runs',
        'name':'train',
        'epochs':20,
        'lr':1e-3,
        'batch_size':8,
        'num_workers':8,
        'weight_decay':1e-5,
        'patience':10,
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
        num_workers=cfg['num_workers'], collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn
    )

    # 加载预训练模型
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)

    # 获取分类头的输入通道数
    in_channels = [512, 1024, 512, 256, 256, 256]  # For VGG16 backbone in SSD300

    # 替换分类头以匹配类别数
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=[4, 6, 6, 6, 4, 4],  # Default anchors for SSD300
        num_classes=num_classes,
    )

    model.to(cfg['device'])

    # 定义优化器和学习率调度器
    scaler = GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup']*len(train_loader) if cfg['warmup'] else 0
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)

    # 开始训练
    stats = [] #评估结果列表
    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for i, (images, targets) in enumerate(pbar):
            images = list(image.to(cfg['device'], non_blocking=True) for image in images)
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

        coco_eval = evaluate(model, val_loader, cfg['device'])
        mAP = coco_eval.stats[0] if coco_eval else 0.0

        early_stopper.update(mAP, model)
        if early_stopper.early_stop:
            print(f'Early stopping with mAP {mAP:.6f}')
            break

    print("Training finished.")


if __name__ == '__main__':
    train(
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=20,
        lr=1e-3,
        warmup=1,
        num_workers=4
    )