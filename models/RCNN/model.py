import torch
import torch.nn as nn
import torchvision


class RCNN(nn.Module):
    """R-CNN 模型，使用 ResNet-50 作为骨干网络。"""

    def __init__(self, num_classes, unfreeze_layers=0):
        super(RCNN, self).__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        # 特征提取器
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # 默认冻结所有骨干网络层
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 根据 unfreeze_layers 解冻 ResNet 的最后几层
        if unfreeze_layers > 0:
            for i in range(1, unfreeze_layers + 1):
                layer_to_unfreeze = list(self.feature_extractor)[-i]
                print(f"Unfreezing backbone layer: layer{5 - i}")
                for param in layer_to_unfreeze.parameters():
                    param.requires_grad = True

        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 2048

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes + 1)  # 类别数 + 1个背景类
        )

        # 边界框回归器
        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes * 4)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        class_scores = self.classifier(features)
        bbox_deltas = self.bbox_regressor(features)
        return class_scores, bbox_deltas