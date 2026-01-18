#https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/segmentation/replknet.py

import torch.nn as nn
import torch.nn.functional as F


def fuse_bn(conv, bn):
    """融合 Conv 和 BN 的权重"""
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """
    RepLKNet 的核心卷积层：
    训练时：大核卷积 + 小核卷积 (3x3) + 旁路 (Identity)
    推理时：融合为一个大核卷积
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 small_kernel=3, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # 此时是否已经把小核融合进了大核
        self.merged = small_kernel_merged

        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                                         bias=True)
        else:
            # 大核分支 (Large Kernel)
            self.lk_origin = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                                       bias=False)
            self.lk_bn = nn.BatchNorm2d(out_channels)

            # 小核分支 (Small Kernel, usually 3x3)
            # 只有当 kernel_size > small_kernel 时才需要小核辅助
            if kernel_size > small_kernel:
                self.small_conv = nn.Conv2d(in_channels, out_channels, small_kernel, stride, small_kernel // 2,
                                            groups=groups, bias=False)
                self.small_bn = nn.BatchNorm2d(out_channels)
            else:
                self.small_conv = None

    def forward(self, inputs):
        if self.merged:
            return self.lkb_reparam(inputs)

        # 大核输出
        out = self.lk_bn(self.lk_origin(inputs))

        # 加上小核输出
        if hasattr(self, 'small_conv') and self.small_conv is not None:
            out += self.small_bn(self.small_conv(inputs))

        return out

    def get_equivalent_kernel_bias(self):
        """核心逻辑：将所有分支融合为一个等效的 Kernel 和 Bias"""
        eq_k, eq_b = fuse_bn(self.lk_origin, self.lk_bn)

        if hasattr(self, 'small_conv') and self.small_conv is not None:
            small_k, small_b = fuse_bn(self.small_conv, self.small_bn)
            # 将小核填充到和大核一样大，然后相加
            eq_b += small_b
            # 计算填充量
            pad = (self.kernel_size - self.small_kernel) // 2
            eq_k += F.pad(small_k, [pad, pad, pad, pad])

        return eq_k, eq_b

    def switch_to_deploy(self):
        """切换到推理模式：融合参数"""
        if self.merged:
            return

        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(self.lk_origin.in_channels, self.lk_origin.out_channels,
                                     self.lk_origin.kernel_size, self.lk_origin.stride,
                                     self.lk_origin.kernel_size // 2, groups=self.lk_origin.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b

        # 删除不需要的分支以节省内存
        self.__delattr__('lk_origin')
        self.__delattr__('lk_bn')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
            self.__delattr__('small_bn')
        self.merged = True


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv',
                      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(out_channels))
    result.add_module('relu', nn.ReLU())  # RepLKNet 原版是用 ReLU 或 GELU
    return result


class RepLKBlock(nn.Module):
    """
    RepLKNet 的基础模块。
    结构：1x1 Conv (Pre) -> ReparamLargeKernelConv -> 1x1 Conv (Post)
    Residual: Input + Output
    """

    def __init__(self, c1, c2, n=1, k=31, s=1, small_kernel=5):
        """
        c1: input channels
        c2: output channels
        n:  number of repeats (YOLO standard arg, handled externally usually)
        k:  large kernel size (e.g., 13, 31)
        s:  stride
        """
        super().__init__()
        # 为了适配 YOLO 的 parse_model，我们只在这里实现单个 Block
        # 如果 n > 1，YOLO 会重复堆叠这个 Block

        # 内部 bottleneck 通道数（RepLKNet 通常保持通道数不变或按比例）
        # 这里简化为 c2，因为 RepLKBlock 通常用于大核提取特征
        mid_c = c2

        self.pre_conv = conv_bn_relu(c1, mid_c, 1, 1, 0, 1)
        self.lk_conv = ReparamLargeKernelConv(mid_c, mid_c, k, s, groups=mid_c, small_kernel=small_kernel)
        self.post_conv = conv_bn_relu(mid_c, c2, 1, 1, 0, 1)

        # 激活函数后处理? 原版在 lk_conv 后接了 BN + Activation
        # 为了通用性，我们在 ReparamLargeKernelConv 内部其实只做了线性融合，
        # 所以这里在 lk_conv 输出后，RepLKNet 论文推荐是 BN -> GELU -> 1x1
        # 但这里我们简化为标准的 Bottleneck 结构

        self.activation = nn.ReLU()  # 或者 nn.SiLU() 适配 YOLO

    def forward(self, x):
        identity = x

        out = self.pre_conv(x)
        out = self.lk_conv(out)
        out = self.activation(out)  # Large Kernel 后的激活
        out = self.post_conv(out)

        if x.shape == out.shape:
            out += identity
        return out