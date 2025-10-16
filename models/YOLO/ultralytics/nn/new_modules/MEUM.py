#https://dl.acm.org/doi/full/10.1145/3729706.3729792#sec-2-2
#由Gemini阅读论文后生成

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancer(nn.Module):
    """
    边缘增强器模块 (EdgeEnhancer Block)。
    根据论文中的描述，该模块通过计算特征图与其平均池化版本的差值来提取边缘特征，
    然后通过 1x1 卷积和 Sigmoid 激活函数来增强这些特征。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 平均池化层用于提取背景或平滑区域信息
        # 论文未指定核大小，这里使用 3x3 作为一个合理的默认值
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 1x1 卷积和 Sigmoid 用于增强提取出的边缘特征
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x: 输入特征图。
        """
        # 公式(3): Edge = F - AvgPool(F)
        # 计算特征图与其平滑版本之间的差异以提取边缘
        edge = x - self.avg_pool(x)

        # 公式(4): EnhancedEdge = σ(Conv1x1(Edge)) [cite: 142]
        # 增强提取到的边缘信息
        enhanced_edge = self.sigmoid(self.conv(edge))

        return enhanced_edge


class MEUM(nn.Module):
    """
    多尺度边缘感知上采样模块 (Multi-scale Edge-aware Upsampling Module)。
    该模块是为 U-Net 的解码器设计的，旨在替代传统的上采样方法（如转置卷积）。
    """

    def __init__(self, channels: int):
        super().__init__()
        # MEUM 模块由两个关键部分组成：一个上采样层和一个多尺度边缘增强模块(MEEM)
        # 上采样层在 forward 中通过双线性插值实现

        # 多尺度边缘增强模块 (MEEM)
        # MEEM 首先对输入特征图应用 1x1 卷积和 Sigmoid 激活
        self.meem_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.meem_sigmoid = nn.Sigmoid()

        # MEEM 内部使用 EdgeEnhancer 块来处理边缘 [cite: 125, 126]
        self.edge_enhancer = EdgeEnhancer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x: 来自解码器前一层的低分辨率特征图。
        """
        # 步骤 1: 上采样层
        # 使用双线性插值将特征图的分辨率提高 2 倍
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # 步骤 2: 多尺度边缘增强模块 (MEEM)
        # 对上采样后的特征图进行非线性变换
        x_transformed = self.meem_sigmoid(self.meem_conv(x_up))

        # 使用 EdgeEnhancer 块提取并增强边缘特征
        enhanced_edge = self.edge_enhancer(x_transformed)

        # 步骤 3: 最终融合
        # 将增强的边缘特征添加回原始上采样特征图，以恢复细节
        output = x_up + enhanced_edge

        return output