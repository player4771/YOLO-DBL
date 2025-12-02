# https://github.com/moskomule/senet.pytorch/tree/master

from torch import nn


class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #自适应平均池化，输出固定为B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False), #把channel的最后一维//r
            nn.ReLU(inplace=True), #非线性
            nn.Linear(in_channels // reduction, in_channels, bias=False), #把channel的最后一维缩放回去
            nn.Sigmoid() #挤压到0-1之间，防止梯度消失或爆炸。注：不是线性缩放也不是标准化，只是基于sigmoid函数的挤压。
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #平均池化 -> (B,C,1,1) -> (B,C)
        y = self.fc(y).view(b, c, 1, 1) #用fc层处理之后缩放回(B,C,1,1)
        return x * y.expand_as(x) #y为由x计算出的权重。x乘权重后扩展到x的形状(B,C,H,W)
