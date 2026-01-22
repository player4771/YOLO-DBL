import yaml
import torch
import torchvision
from pathlib import Path

torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

#import sys
#sys.path.append('/root/project/Paper2/')

from backbone import ResNetBackbone
from global_utils import ATransforms, Trainer, default_val, default_detect


def create_model(backbone:str='vgg16', num_classes:int=4, weights:str=None): # -> model
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
        raise "Invalid backbone, requires 'vgg16' or 'resnet50'."

    if weights:
        model.load_state_dict(torch.load(weights))

    return model

def train(**kwargs):
    cfg = { #default args
        'backbone':'vgg16',
        'data':"E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        'dataset': {}, #避免类型检查警告
        'project':'./runs',
        'name':'train',
        'epochs':100,
        'lr':1e-2,
        'lf':1e-2,
        'batch':4,
        'workers':4,
        'weight_decay':1e-5,
        'patience':10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'warmup': 3,
        'img_size': 640,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg['dataset'] = yaml.load(infile, Loader=yaml.FullLoader)

    num_classes = cfg['dataset']['nc'] + 1 # +1 是因为 SSD 需要一个背景类
    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)

    trainer = Trainer(
        model=model,
        transform_train=ATransforms(is_train=True, size=cfg['img_size']),
        transform_val=ATransforms(is_train=False, size=cfg['img_size']),
        **cfg
    )
    trainer.start_training()

def val():
    input_dir = Path('./runs/train6')
    model = create_model(weights=str(input_dir / 'best.pth'))
    transform = ATransforms(is_train=False, size=640)
    default_val(model, input_dir, transform)

def detect():
    model = create_model(weights='./runs/train6/best.pth')
    transform = ATransforms(is_train=False, size=640)
    default_detect(
        model,
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        input_="E:/Projects/Datasets/example/sample_v4_1.jpg",
        project='./runs',
        transform=transform,
        conf_thres=0.5,
    )


if __name__ == '__main__':
    train()
    #val()
    #detect()