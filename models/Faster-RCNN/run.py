import yaml
import torch
import torchvision
from pathlib import Path
torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

import sys
sys.path.append('/root/project/Paper2/')

from global_utils import Trainer, ATransforms, default_val, default_detect


def create_model(backbone='resnet50', num_classes=4, weights=None, **kwargs):
    models = {
        'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'mobilenet320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    }

    model = models[backbone](weights='DEFAULT', **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    if weights:
        model.load_state_dict(torch.load(weights), strict=False)
    return model

def train(**kwargs):
    cfg = {
        'backbone': 'resnet50',
        'data': "E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        'project': './runs',
        'name': 'train',
        'dataset': {},
        'epochs': 100,
        'lr': 1e-3,
        'lf': 1e-2,
        'batch': 8,
        'workers': 4,
        'weight_decay': 1e-5,
        'patience': 10,
        'delta': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'warmup': 3,
        'img_size': 640,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg['dataset'] = yaml.load(infile, Loader=yaml.FullLoader)

    num_classes = cfg['dataset']['nc'] + 1
    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)

    trainer = Trainer(
        model=model,
        transform_train=ATransforms(is_train=True, size=cfg['img_size']),
        transform_val=ATransforms(is_train=False, size=cfg['img_size']),
        **cfg
    )
    trainer.start_training()

def val():
    input_dir = Path('./runs/train3')
    model = create_model(weights=str(input_dir / 'best.pth'))
    transform = ATransforms(is_train=False, size=640)
    default_val(
        model,
        input_dir,
        transform,
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        batch=8,
        workers=4,
    )

def detect():
    model = create_model(weights='./runs/train3/best.pth')
    transform = ATransforms(is_train=False, size=640)
    default_detect(
        model,
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        input_="E:/Projects/Datasets/example/sample_v4_1.jpg",
        project='./runs',
        transform=transform,
        conf_thres=0.1,
    )


if __name__ == '__main__':
    #train()
    #val()
    detect()