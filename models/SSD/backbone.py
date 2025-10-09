import torchvision
import torch.nn as nn
from collections import OrderedDict

class ResNetBackbone(nn.Module):
    """用于SSD的ResNet50骨干网络，包含额外的特征提取层。"""

    def __init__(self):
        super().__init__()

        # 加载预训练的ResNet50模型
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        # 我们需要从中间层提取特征，所以只保留到 layer3
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])

        # 定义额外的卷积层以生成用于检测的更小尺寸的特征图
        self.extra_layers = nn.ModuleList([
            nn.Sequential(  # 从 layer3 的输出 (1024) 开始
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        ])

        # 初始化额外层的权重
        for m in self.extra_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 用于从ResNet的特定层提取特征的钩子(hooks)
        self.out_channels = []  # 记录输出通道
        self.outputs = {}
        # 截取 layer2 和 layer3 的输出
        self.feature_extractor[-2].register_forward_hook(self.get_hook('layer2'))
        self.feature_extractor[-1].register_forward_hook(self.get_hook('layer3'))

    def get_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output

        return hook

    def forward(self, x):
        # 使用 OrderedDict 来保证特征图的顺序
        features = OrderedDict()
        self.outputs = {}

        # 经过ResNet主干
        self.feature_extractor(x)
        features['0'] = self.outputs['layer2']  # 第一个特征图来自 layer2

        # extra_input 从 layer3 的输出开始
        extra_input = self.outputs['layer3']
        features['1'] = extra_input

        # 经过额外的卷积层
        for i, layer in enumerate(self.extra_layers):
            extra_input = layer(extra_input)
            features[str(i + 2)] = extra_input

        return features