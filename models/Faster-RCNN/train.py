import yaml
import torch
import torchvision
torch.hub.set_dir('./') #修改缓存路径
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from global_utils import Trainer, ATransforms


def create_model(backbone='resnet50', num_classes=4, **kwargs):
    models = {
        'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'resnet50v2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'mobilenet320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    }

    model = models[backbone](weights='DEFAULT', **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train(**kwargs):
    cfg = {
        'backbone': 'resnet50',
        'data': None,
        'project': './runs',
        'name': 'train',
        'dataset': {},
        'epochs': 20,
        'lr': 1e-3,
        'lf': 1e-2,
        'batch_size': 8,
        'num_workers': 8,
        'weight_decay': 1e-5,
        'patience': 5,
        'delta': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'warmup_epochs': 0,  # 预热epoch数，None为禁用
        'img_size': 640,
    }
    cfg.update(kwargs)

    with open(cfg['data'], 'r') as infile:
        cfg['dataset'] = yaml.load(infile, Loader=yaml.FullLoader)

    num_classes = cfg['dataset']['nc'] + 1
    model = create_model(backbone=cfg['backbone'], num_classes=num_classes)

    trainer = Trainer(
        model=model,
        collate_fn=collate_fn,
        transform_train=ATransforms(is_train=True, size=cfg['img_size']),
        transform_val=ATransforms(is_train=False, size=cfg['img_size']),
        **cfg
    )
    trainer.start_training()


if __name__ == '__main__':
    train(
        backbone='resnet50v2',
        data="E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        project="./runs",
        epochs=100,
        patience=7,
        lr=1e-3,
        warmup_epochs=1,
        batch_size=8,
        num_workers=4,
        img_size=300,
    )