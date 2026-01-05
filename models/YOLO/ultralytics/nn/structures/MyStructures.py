import torch
from torch import nn

class ExtractLayer(nn.Module):
    """
    用于某些输出包含多个张量的元组/列表的模块。\n
    可以从输出中提取其中一个张量以供其余模块使用。
    Arguments:
        from_index: 要提取输入中的第几个张量
    """
    def __init__(self, from_index:int=0):
        super().__init__()
        self.from_index = from_index

    def forward(self, x):
        return x[self.from_index]

