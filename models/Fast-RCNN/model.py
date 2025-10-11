import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FastRCNN(nn.Module):
    def __init__(self, num_classes:int=4):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=(7, 7), spatial_scale=1.0 / 32.0, sampling_ratio=-1, aligned=True)

        # 通过将 4096 维度降低到 1024，大幅减少参数量
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024), # 4096 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024), # 4096 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分类器和回归器的输入维度也需要相应修改
        self.cls_score = nn.Linear(1024, self.num_classes)
        self.bbox_pred = nn.Linear(1024, 4)

    def forward(self, images, rois):
        features = self.backbone(images)
        roi_features = self.roi_align(features, rois)
        x = self.head(roi_features)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FastRCNNFPN(nn.Module):
    def __init__(self, num_classes: int):
        super(FastRCNNFPN, self).__init__()
        self.num_classes = num_classes

        # 加载带有FPN的ResNet-50作为骨干网络
        self.backbone = resnet_fpn_backbone(backbone_name='resnet50',
                                            weights=torchvision.models.ResNet50_Weights.DEFAULT)

        # 多尺度RoI对齐层，用于从FPN的不同层级提取特征
        self.roi_align = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # 对应FPN输出的P2,P3,P4,P5
            output_size=7,
            sampling_ratio=2
        )

        # 轻量化的检测头，集成了BatchNorm以提升稳定性
        # FPN的输出通道数为256
        fpn_out_channels = 256
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fpn_out_channels * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分类器，用于预测RoI的类别
        self.cls_score = nn.Linear(1024, self.num_classes)

        # 回归器，用于预测边界框的偏移量
        self.bbox_pred = nn.Linear(1024, self.num_classes * 4)

    def forward(self, images, rois):
        # 从骨干网络和FPN获取多尺度特征图
        # 输入: images (Tensor[N, C, H, W])
        # 输出: features (OrderedDict[str, Tensor])
        features = self.backbone(images)

        # 将RoIs按批次索引拆分为列表
        # rois: [K, 5] -> (batch_idx, x1, y1, x2, y2)
        rois_list = []
        for i in range(images.shape[0]):
            rois_for_image = rois[rois[:, 0] == i, 1:]
            rois_list.append(rois_for_image)

        # 获取每张图片的尺寸
        image_shapes = [img.shape[-2:] for img in images]

        # 在多尺度特征图上进行RoI对齐
        # 输入: features, rois_list, image_shapes
        # 输出: roi_features (Tensor[K, C, output_size, output_size])
        roi_features = self.roi_align(features, rois_list, image_shapes)

        # 通过检测头网络
        x = self.head(roi_features)

        # 分别计算分类得分和边界框回归偏移量
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
