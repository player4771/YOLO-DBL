import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIAlign


class FastRCNN(nn.Module):
    def __init__(self, num_classes=81, pretrained=True):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0 / 32.0, sampling_ratio=-1)

        # --- 核心优化：创建一个更轻量的头部网络 ---
        # 通过将 4096 维度降低到 1024，大幅减少参数量
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),   # <-- 从 4096 改为 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024), # <-- 从 4096 改为 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分类器和回归器的输入维度也需要相应修改
        self.cls_score = nn.Linear(1024, self.num_classes)
        self.bbox_pred = nn.Linear(1024, self.num_classes * 4)

    def forward(self, images, rois):
        features = self.backbone(images)
        roi_features = self.roi_align(features, rois)
        x = self.head(roi_features)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas