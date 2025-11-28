import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareAttention(nn.Module):
    """
    可以正常使用，对精度没有明显负面影响。
    """
    def __init__(self, channels, reduction=16, ksize=7):
        super().__init__()
        self.channels = channels
        # 空间注意力：输入4通道 (avgX, maxX, avgG, maxG)
        self.spatial = nn.Conv2d(4, 1, ksize, padding=ksize//2, bias=True)
        # 通道注意力：MLP
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        # c_gain和s_gain分别为通道和空间注意力的增益
        self.c_gain = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.s_gain = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # 注册 Sobel 作为 buffer，随设备/精度移动
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device='cpu')/4.0
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device='cpu')/4.0
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))

    def forward(self, x):  # x: BCHW
        B, C, H, W = x.shape

        #分离出一个x的副本，否则反向传播时会产生负面效果
        x_edge = x.detach()
        # depthwise Sobel（自动广播到通道）
        kx = self.kx.to(device=x.device, dtype=x.dtype).repeat(C,1,1,1)
        ky = self.ky.to(device=x.device, dtype=x.dtype).repeat(C,1,1,1)
        gx = F.conv2d(x_edge, kx, padding=1, groups=C)
        gy = F.conv2d(x_edge, ky, padding=1, groups=C)
        # g: 边缘幅值特征图. g[:, :, h, w]表示(h,w)处在对应通道上是不是边缘、边缘有多强
        # 即边缘先验, 从而告诉后面的两个注意力哪里该被增强
        g  = torch.sqrt(gx*gx + gy*gy + 1e-12) # B,C,H,W

        # --- Channel Attention ---
        # 在空间维度对g做平均. 每个通道一个标量，表示这个通道整体的边缘能量
        c_vec = g.mean(dim=(-1,-2)) # B,C
        # c: 通道注意力权重, 代表了某个通道在边缘强度上的重要程度
        # 若某个通道边缘更多/更重要，则值就会更高
        c = torch.sigmoid(self.mlp(c_vec)).view(B,C,1,1) # 可以广播到BCHW

        # --- Spatial Attention ---
        avg_x = x.mean(dim=1, keepdim=True) # B,1,H,W, 代表激活强度
        max_x = x.amax(dim=1, keepdim=True) # B,1,H,W, 代表激活强度
        avg_g = g.mean(dim=1, keepdim=True) # B,1,H,W, 代表边缘强度
        max_g = g.amax(dim=1, keepdim=True) # B,1,H,W, 代表边缘强度
        s_in = torch.cat([avg_x, max_x, avg_g, max_g], dim=1)  # B,4,H,W
        # s: 单通道空间注意力图. 代表某个像素在空间上的重要程度
        # 若某处的特征强且具有明显边缘，则值就高
        s = torch.sigmoid(self.spatial(s_in)) # B,1,H,W

        # 残差增益, y = x * weight, weight就是这个模块要计算出的权重
        # 两个gain是对注意力权重的增益，也是可学习参数，可以学习出合适的大小
        y = x * (1 + self.c_gain(c)) * (1 + self.s_gain(s))
        return y


class EdgeAwareAttentionV2(nn.Module):
    """
    ChatGPT5 Pro根据v1做出的改进版，太屌了。\n
    改进点：
      - 多算子（Sobel/Scharr/Prewitt）可学习边缘核 + 动态选择（gate_mlp + softmax）
      - Charbonnier 边缘幅值（更抗噪）
      - 自适应增益：alpha(按图自适应的空间增益, scalar 或 map)，beta(按通道自适应的通道增益)
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        ksize: int = 7,
        kernel_bank=("sobel", "scharr", "prewitt"),  # 多算子银行
        learnable_k: bool = True,
        kernel_size: int = 3,
        charbonnier_eps: float = 1e-3,              # (3) 更鲁棒的范数
        alpha_mode: str = "scalar"                  # "scalar" 或 "map"
    ):
        super().__init__()
        assert kernel_size == 3, "此示例实现了 3x3 核，如需其它尺寸可按同样思路扩展。"
        assert alpha_mode in ("scalar", "map")
        self.channels = channels
        self.num_k = len(kernel_bank)
        self.kernel_size = kernel_size
        self.charbonnier_eps = charbonnier_eps
        self.alpha_mode = alpha_mode

        # --- 空间注意力：与原版一致（输入 avg/max of x 与 g） ---
        self.spatial = nn.Conv2d(4, 1, ksize, padding=ksize // 2, bias=True)

        # --- 通道注意力 MLP（输入为通道级边缘统计） ---
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

        # --- (2) 多算子可学习边缘核 ---
        kx0, ky0 = [], []
        for name in kernel_bank:
            kx, ky = self._get_kernel_pair(name)  # 3x3
            kx0.append(kx)
            ky0.append(ky)
        kx0 = torch.stack(kx0)[:, None, :, :]  # (N,1,3,3)
        ky0 = torch.stack(ky0)[:, None, :, :]  # (N,1,3,3)

        if learnable_k:
            self.kx = nn.Parameter(kx0)  # (N,1,3,3)
            self.ky = nn.Parameter(ky0)
        else:
            self.register_buffer("kx", kx0)       # 作为 buffer 固定
            self.register_buffer("ky", ky0)

        # --- (2) 动态核选择 gate：根据图像边缘统计为每个算子分配权重 ---
        gate_h = max(8, self.num_k * 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.num_k, gate_h),
            nn.ReLU(inplace=True),
            nn.Linear(gate_h, self.num_k)  # softmax 之前的 logits
        )

        # --- (6) 自适应增益 ---
        # alpha：基于全局边缘统计（mean/max） -> 标量；或 1x1 conv -> 空间图
        if self.alpha_mode == "scalar":
            self.alpha_head = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1)
            )
        else:  # "map"
            self.alpha_conv = nn.Conv2d(4, 1, kernel_size=1, bias=True)

        # beta：基于通道边缘统计（B,C） -> 逐通道非负增益
        self.beta_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    # ----------------------------- 辅助函数 -----------------------------
    @staticmethod
    def _normalize_k(k):  # 零均值 + L1 归一，稳定训练
        k = k - k.mean(dim=(2, 3), keepdim=True)
        l1 = k.abs().sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return k / l1

    @staticmethod
    def _get_kernel_pair(name:str):
        if name.lower() == "sobel":
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32) / 4.0
            ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32) / 4.0
        elif name.lower() == "scharr":
            kx = torch.tensor([[3,0,-3],[10,0,-10],[3,0,-3]], dtype=torch.float32) / 16.0
            ky = torch.tensor([[3,10,3],[0,0,0],[-3,-10,-3]], dtype=torch.float32) / 16.0
        elif name.lower() == "prewitt":
            kx = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=torch.float32) / 3.0
            ky = torch.tensor([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=torch.float32) / 3.0
        elif name.lower() == "log": #Laplacian of Gaussian
            kx = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            ky = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        elif name.lower() == "kirsch":
            kx = torch.tensor([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=torch.float32)  # First direction
            ky = torch.tensor([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=torch.float32)  # Second direction
        elif name.lower() == "roberts":
            kx = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)  # Diagonal gradient in X-direction
            ky = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)  # Diagonal gradient in Y-direction
        elif name.lower() == "prewitt_alt": # slightly different weights
            kx = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32)
            ky = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
        elif name.lower() == "sobel_alt": # Modified Sobel operator
            kx = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            ky = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown kernel name: {name}")
        return kx, ky

    def forward(self, x):
        B, C, H, W = x.shape

        # -------- (2)+(3) 多算子 + Charbonnier 计算边缘幅值 --------
        # 归一化核：零均值 + L1 归一，保证稳定 & 保持“微分”性质
        kx = self._normalize_k(self.kx).to(device=x.device, dtype=x.dtype)  # (N,1,3,3)
        ky = self._normalize_k(self.ky).to(device=x.device, dtype=x.dtype)

        # depthwise 卷积：每个通道分别与 N 个核做卷积 => (B, C*N, H, W)
        pad = self.kernel_size // 2
        weight_x = kx.repeat(C, 1, 1, 1)  # (N*C,1,3,3)
        weight_y = ky.repeat(C, 1, 1, 1)

        gx = F.conv2d(x, weight_x, padding=pad, groups=C)  # (B, C*N, H, W)
        gy = F.conv2d(x, weight_y, padding=pad, groups=C)

        # 重塑为 (B, C, N, H, W)
        gx = gx.view(B, C, self.num_k, H, W)
        gy = gy.view(B, C, self.num_k, H, W)

        # Charbonnier 边缘幅值（更抗噪）
        g_bank = torch.sqrt(gx * gx + gy * gy + self.charbonnier_eps * self.charbonnier_eps)  # (B,C,N,H,W)

        # 动态核选择：对 N 个算子 softmax 加权求和 => g: (B,C,H,W)
        # 描述子：按 (C,H,W) 平均得到每个算子的全局响应 (B,N)
        gate_desc = g_bank.mean(dim=(1, 3, 4))  # (B,N)
        gate_w = F.softmax(self.gate_mlp(gate_desc), dim=-1)  # (B,N)
        gate_w = gate_w[:, None, :, None, None]               # (B,1,N,1,1)
        g = (g_bank * gate_w).sum(dim=2)                      # (B,C,H,W)

        # -------- 通道注意力（仍然由边缘统计驱动） --------
        c_vec = g.mean(dim=(-1, -2))                          # (B,C)
        c = torch.sigmoid(self.mlp(c_vec)).view(B, C, 1, 1)   # (B,C,1,1)

        # -------- 空间注意力（与原版一致，用 g 替代旧的边缘） --------
        avg_x = x.mean(1, keepdim=True)
        max_x = x.amax(1, keepdim=True)
        avg_g = g.mean(1, keepdim=True)
        max_g = g.amax(1, keepdim=True)
        s_in = torch.cat([avg_x, max_x, avg_g, max_g], dim=1)  # (B,4,H,W)
        s = torch.sigmoid(self.spatial(s_in))                  # (B,1,H,W)

        # -------- (6) 自适应增益 --------
        # alpha：非负（softplus），可选 scalar / map
        if self.alpha_mode == "scalar":
            stats = torch.stack(
                [g.mean(dim=(1, 2, 3)), g.amax(dim=(1, 2, 3))],
                dim=1
            )  # (B,2)
            alpha = F.softplus(self.alpha_head(stats)).view(B, 1, 1, 1)  # (B,1,1,1)
        else:
            alpha = F.softplus(self.alpha_conv(s_in))  # (B,1,H,W)

        # beta：逐通道非负增益（对 c 的强度进行自适应缩放）
        beta = F.softplus(self.beta_mlp(c_vec)).view(B, C, 1, 1)  # (B,C,1,1)

        # 残差式增强：保持与原模块的形式兼容
        y = x * (1 + alpha * s) * (1 + beta * c)
        return y


if __name__ == '__main__':
    import joblib
    from global_utils import plot_feature_maps

    layer_indexes = (26, 31, 36)
    results = joblib.load(rf"E:\Projects\PyCharm\Paper2\global_utils\cache\{hash(layer_indexes)}.cache")

    input = results[layer_indexes[0]]['output']
    channels = input.shape[-3]
    assert isinstance(input, torch.Tensor), TypeError(f'Invalid type: {type(input)}, expected torch.Tensor')

    model = EdgeAwareAttentionV2(channels=channels, ksize=3)
    with torch.no_grad():
        output = model.to('cuda').eval()(input.to('cuda'))

    plot_feature_maps(input, output)