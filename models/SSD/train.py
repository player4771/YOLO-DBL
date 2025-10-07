import os
import yaml
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark=True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from src import Backbone, SSD300
from global_utils import EarlyStopping, YoloDataset, AlbumentationsTransform, find_new_dir, evaluate, convert_to_coco_api

def create_model(backbone:str='vgg16', weights:str=None, num_classes:int=21):# -> model
    if backbone == "vgg16":
        model = torchvision.models.detection.ssd300_vgg16(
            weights=weights if weights else torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1)
        # 替换分类头以匹配类别数
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=[512, 1024, 512, 256, 256, 256],  # For VGG16 backbone in SSD300
            num_anchors=[4, 6, 6, 6, 4, 4],  # Default anchors for SSD300
            num_classes=num_classes,
        )
    elif backbone == "resnet50": #暂时无法实现
        model = SSD300(backbone=Backbone(), num_classes=num_classes)
        pre_model_dict = torch.load(weights if weights else "./src/nvidia_ssdpyt_amp_200703.pt", map_location='cpu')
        pre_weights_dict = pre_model_dict["model"]
        # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
        del_conf_loc_dict = {}
        for k, v in pre_weights_dict.items():
            split_key = k.split(".")
            if "conf" in split_key:
                continue
            del_conf_loc_dict.update({k: v})
        model.load_state_dict(del_conf_loc_dict, strict=False)
    else:
        raise "Invalid backbone, required 'vgg16' or 'resnet50'."

    return model


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
    output_dir = find_new_dir(Path(cfg['project'], cfg['name']))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir/'args.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile)

    data_path = Path(cfg['data']).parent
    train_img_path = data_path / cfg['train']
    val_img_path = data_path / cfg['val']
    train_label_path = train_img_path.parent / 'labels'
    val_label_path = val_img_path.parent / 'labels'
    results_file = output_dir/'results.csv'

    # 创建 Dataset
    dataset_train = YoloDataset(
        img_dir=str(train_img_path), label_dir=str(train_label_path),
        transform=AlbumentationsTransform(is_train=True)
    )
    dataset_val = YoloDataset(
        img_dir=str(val_img_path), label_dir=str(val_label_path),
        transform=AlbumentationsTransform(is_train=False)
    )

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn, persistent_workers=True
    )

    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)
    model.to(cfg['device'])

    scaler = torch.amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * cfg['lf'])
    early_stopper = EarlyStopping(patience=cfg['patience'], verbose=True, path=str(output_dir/'best.pth'))
    warmup_iters = cfg['warmup']*len(train_loader) if cfg['warmup'] else 0
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters)
    coco_gt = convert_to_coco_api(val_loader.dataset)

    for epoch in range(cfg['epochs']):
        model.train()

        pbar = tqdm(train_loader, desc=f"Train[{epoch}]")
        for (images, targets) in pbar:
            #images = list(image.to(cfg['device'], non_blocking=True) for image in images)
            images = torch.stack(images, dim=0).to(device=cfg['device'], non_blocking=True)
            targets = [{k: v.to(cfg['device'], non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=cfg['device']):
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
        batch_size=8,
        num_workers=4
    )