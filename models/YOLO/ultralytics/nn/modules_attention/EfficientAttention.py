import torch
from torch import nn
from torch.nn import functional as f


class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels:int, key_channels:int=8, head_count:int=None, value_channels:int=None):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count if head_count is not None else in_channels
        self.value_channels = value_channels if value_channels is not None else in_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, self.value_channels, 1)
        self.reprojection = nn.Conv2d(self.value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class EfficientAttention_YOLO(nn.Module):
    """
    YOLO 专用封装类
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数 (EfficientAttention 是残差结构，通常要求 c1 == c2)
        head_count (int): 注意力头数
        key_channels (int): 键值通道数 (必须能被 head_count 整除)
    """

    def __init__(self, c1, c2, head_count=8, key_channels=64):
        super().__init__()
        # EfficientAttention 包含 x + attention，要求输入输出维度一致
        # 如果 YOLO 传入的 c2 != c1，我们需要警告或处理，这里强制保持一致
        assert c1 == c2, f"EfficientAttention expects same input/output channels, but got c1={c1}, c2={c2}"

        # 修正原代码潜在的除零 bug：确保 key_channels >= head_count
        if key_channels < head_count:
            key_channels = head_count * 4  # 自动调整为一个合理值

        self.att = EfficientAttention(
            in_channels=c1,
            key_channels=key_channels,
            head_count=head_count,
            value_channels=c1  # 通常 value_channels 保持与输入一致
        )

    def forward(self, x):
        return self.att(x)