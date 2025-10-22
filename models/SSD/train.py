import yaml
import torch
import torchvision

torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark=True
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

from backbone import ResNetBackbone
from global_utils import YoloDataset, AlbumentationsTransform, Trainer

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
        anchor_generator = DefaultBoxGenerator(
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
    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)
    transform_train = AlbumentationsTransform(is_train=True, size=300)
    transform_val = AlbumentationsTransform(is_train=False, size=300)


    trainer = Trainer(model=model,
                      dataset_class=YoloDataset,
                      collate_fn=collate_fn,
                      transform_train=transform_train,
                      transform_val=transform_val,
                      **cfg
                      )
    trainer.start_training()


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

