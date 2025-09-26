# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIAlign


class FastRCNN(nn.Module):
    def __init__(self, num_classes=81, pretrained=True):
        """
        Args:
            num_classes (int): 类别数 (包含背景)
            pretrained (bool): 是否加载预训练的ResNet50
        """
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        # 加载ResNet50主干网络
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # 去掉最后的avgpool和fc层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # RoI Align层
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0 / 32.0, sampling_ratio=-1)

        # 头部网络
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分类器分支
        self.cls_score = nn.Linear(4096, self.num_classes)

        # 边框回归器分支
        self.bbox_pred = nn.Linear(4096, self.num_classes * 4)

    def forward(self, images, rois):
        """
        Args:
            images (Tensor): 输入图片, shape [N, C, H, W]
            rois (list of Tensors): 区域提议, list of [K, 5], 5为(batch_idx, x1, y1, x2, y2)
        """
        features = self.backbone(images)
        roi_features = self.roi_align(features, rois)
        x = self.head(roi_features)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas