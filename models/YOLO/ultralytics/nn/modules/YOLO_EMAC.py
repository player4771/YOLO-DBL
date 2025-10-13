#https://github.com/wuhan66/YOLO-EMAC/blob/main/yolov-emac/ultralytics/nn/tasks.py#L937

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import Bottleneck, C2f, C3k


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 group_num: int = 16,  # 分组数，默认为16
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

        # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)  # 创建卷积层
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return  x



class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)


class C2f_ScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

def C3k_ScConv(*args, **kwargs):
    """不知为何，此类/函数在YOLO-EMAC的代码中没有声明\n
    此方法在其余代码中出现，但并未真正使用"""
    pass

class C3k2_ScConv(C2f):
    """
    C3k2 模块与 ScConv 结合版本。
    使用 ScConv 替代 Bottleneck 结构，实现结构信息和通道感知增强。
    """
    def __init__(self, c1, c2, n=1, use_c3k=False, e=0.5, g=1, shortcut=True,
                 group_num=16, gate_treshold=0.5,
                 alpha=0.5, squeeze_radio=2, group_size=2, group_kernel_size=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_ScConv(self.c, self.c, 2, shortcut, g, e, group_num, gate_treshold,
                       alpha, squeeze_radio, group_size, group_kernel_size)
            if use_c3k else
            Bottleneck_ScConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# ------------------- 1. DyT(替代归一化) -------------------
class DyT(nn.Module):
    """
    动态变换层, 用来替代 BN/LN 等常规 Norm.
    这里是 [B, N, C] 的写法.
    如果想在 [B, C, H, W] 直接做, 可以改成 2D 广播方式.
    """
    def __init__(self, channels, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([init_alpha], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, N, C]
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta

# ------------------- 2. WindowMHSA(单尺度窗口注意力,可选Flash) -------------------
try:
    from flash_attn.modules.mha import FlashSelfAttention
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class WindowMHSA(nn.Module):
    """
    支持任意 H,W 的窗口注意力:
    1) 先pad到能被window_size整除
    2) 进行窗口attention
    3) 恢复原大小
    """

    def __init__(self, dim, num_heads, window_size=7, use_flash=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_flash = use_flash

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # 视情况添加flash
        # try:
        #     from flash_attn.modules.mha import FlashSelfAttention
        #     self.flash_attn = FlashSelfAttention()
        #     self.use_flash = use_flash
        # except ImportError:
        #     self.use_flash = False

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # 1) 计算需要pad的行列
        H_pad = (ws - (H % ws)) % ws
        W_pad = (ws - (W % ws)) % ws

        if H_pad > 0 or W_pad > 0:
            # 2) 做补零pad: [left, right, top, bottom]
            x = F.pad(x, (0, W_pad, 0, H_pad))

        # 新的 H2, W2 能整除 ws
        H2, W2 = x.shape[2], x.shape[3]
        nWh = H2 // ws
        nWw = W2 // ws

        # 后面逻辑同之前
        x_reshaped = (x.view(B, C, nWh, ws, nWw, ws)
                      .permute(0, 2, 4, 1, 3, 5)
                      .reshape(B * nWh * nWw, C, ws, ws))

        Bwin = x_reshaped.shape[0]
        N = ws * ws
        qkv = self.qkv(x_reshaped).reshape(Bwin, 3 * C, N).transpose(1, 2)
        q, k, v = torch.split(qkv, C, dim=2)

        # 3) Attention (此处简单用普通attention; flash同理)
        q = q.reshape(Bwin, N, self.num_heads, self.head_dim)
        k = k.reshape(Bwin, N, self.num_heads, self.head_dim)
        v = v.reshape(Bwin, N, self.num_heads, self.head_dim)
        attn_scores = torch.einsum("bnhd,bmhd->bnmh", q, k) * self.scale
        attn = attn_scores.softmax(dim=-1)
        out = torch.einsum("bnmh,bmhd->bnhd", attn, v)
        out = out.reshape(Bwin, N, C)

        out = out.transpose(1, 2).reshape(Bwin, C, ws, ws)
        out = (out.view(B, nWh, nWw, C, ws, ws)
               .permute(0, 3, 1, 4, 2, 5)
               .reshape(B, C, H2, W2))
        out = self.proj(out)

        # 4) 去掉多余的 pad
        if H_pad > 0 or W_pad > 0:
            out = out[:, :, :H, :W]

        return out

# ------------------- 3. 并行多尺度注意力-------------------
class MultiScaleAttention(nn.Module):
    """
    并行分支的窗口注意力: window_sizes=(3,5,7), 可选 Flash.
    """
    def __init__(self, dim, num_heads, window_sizes=(3,5,7), use_flash=False):
        super().__init__()
        self.branches = nn.ModuleList([
            WindowMHSA(dim, num_heads, ws, use_flash=use_flash) for ws in window_sizes
        ])
        self.fuse = nn.Conv2d(dim*len(window_sizes), dim, kernel_size=1, bias=False)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)  # [B, C*len(window_sizes), H, W]
        out = self.fuse(out)
        return out

# ------------------- 4. MBlock(多尺度+DyT) -------------------
class MBlock(nn.Module):
    """
    1) pre-Attn DyT
    2) 并行多窗口注意力 + 残差
    3) pre-MLP DyT
    4) MLP + 残差
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2,
                 window_sizes=(3,5,7), use_flash=False):
        super().__init__()
        self.dyt1 = DyT(dim)
        self.attn = MultiScaleAttention(dim, num_heads, window_sizes, use_flash=use_flash)
        self.dyt2 = DyT(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) pre-Attn DyT
        x_flat = x.permute(0,2,3,1).reshape(B, H*W, C)  # => [B, N, C]
        x_flat = self.dyt1(x_flat)                     # => [B, N, C]
        x_dyt1 = x_flat.view(B, H, W, C).permute(0,3,1,2)

        # 2) MultiScaleAttention + resid
        attn_out = self.attn(x_dyt1)
        x = x + attn_out

        # 3) pre-MLP DyT
        x_flat = x.permute(0,2,3,1).reshape(B, H*W, C)
        x_flat = self.dyt2(x_flat)
        x_dyt2 = x_flat.view(B, H, W, C).permute(0,3,1,2)

        # 4) MLP + resid
        mlp_out = self.mlp(x_dyt2)
        x = x + mlp_out
        return x

# ------------------- M2C2f -------------------
class M2C2f(nn.Module):
    """
    - 当 use_attn=True 时, 用 MBlock(多窗口注意力+DyT)
    - 当 use_attn=False 时, 用 C3k(卷积)
    - 其余R-ELAN通道拼接和可选residual逻辑保持一致
    """
    def __init__(self, c1, c2, n=1, use_attn=True, residual=False,
                 mlp_ratio=2.0, e=0.5, g=1, shortcut=True,
                 window_sizes=(3,5,7), use_flash=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 若想对齐flash,可保证 c_//num_heads=32,64等. 这里简单写:
        num_heads = max(1, c_ // 32)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        # 若 residual=True, 则可学习gamma
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if (use_attn and residual) else None

        # 否则就用C3k
        self.m = nn.ModuleList(
            nn.Sequential(*(MBlock(c_, num_heads, mlp_ratio, window_sizes, use_flash)
                            for _ in range(2)))
            if use_attn else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        # R-ELAN 核心: 先做cv1, 然后把每次输出append到列表, 最后cat并用cv2融合
        y = [self.cv1(x)]
        for module in self.m:
            y.append(module(y[-1]))

        out = self.cv2(torch.cat(y, dim=1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out

class eca_layer_triple(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_triple, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_proj = nn.Conv2d(channel, channel, kernel_size=1)

        # Conv1D for (3C) → C
        self.reduce_conv = nn.Conv1d(3, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 三个特征提取
        avg_out = self.avg_pool(x)       # [B, C, 1, 1]
        max_out = self.max_pool(x)       # [B, C, 1, 1]
        conv_out = self.channel_proj(x)  # [B, C, H, W] → 池化
        conv_out = nn.functional.adaptive_avg_pool2d(conv_out, 1)

        # 拼接 → [B, 3, C]
        y = torch.cat([avg_out, max_out, conv_out], dim=1)  # [B, 3C, 1, 1]
        y = y.view(b, 3, c)  # reshape 成 [B, 3, C]，以便 Conv1d 沿通道建模

        # Conv1d 注意力建模
        y = self.reduce_conv(y)  # [B, 1, C]
        y = self.sigmoid(y)

        # reshape 成 [B, C, 1, 1]
        y = y.view(b, c, 1, 1)

        return x * y.expand_as(x)


class C3k2_EAMC(C2f):
    """C3k2 with Triple-Feature ECA Attention"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, eca_k=3):
        super().__init__(c1, c2, n, shortcut, g, e)

        # 替换中间模块为 Bottleneck 或 C3k
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
        self.eca = eca_layer_triple(c2, k_size=eca_k)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))             # 两路分支
        y.extend(m(y[-1]) for m in self.m)            # 多层残差
        out = self.cv2(torch.cat(y, 1))               # 合并特征
        return self.eca(out)                          # 加上 EAMC 通道注意力