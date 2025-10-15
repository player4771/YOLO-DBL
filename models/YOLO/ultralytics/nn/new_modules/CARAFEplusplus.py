import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFEplusplus(nn.Module):
    """
    CARAFE++: Unified Content-Aware ReAssembly of FEatures\n
    https://www.webofscience.com/wos/woscc/full-record/WOS:000836666600017\n
    输入尺寸较小时(如2x64x64x64)速度显著优于CARAFE，但尺寸较大时(如2x64x256x256)显存占用极高，易导致显存不足，且性能也仅有CARAFE的1/4左右。
    """

    def __init__(self,
                 in_channels: int,
                 scale_factor: int,
                 up_down_type: str = 'up',
                 k_encoder: int = 3,
                 k_reassembly: int = 5):
        """
        初始化 CARAFE++ 模块。

        参数:
            in_channels (int): 输入特征图的通道数。
            scale_factor (int): 上采样或下采样的尺度因子。
            up_down_type (str): 操作类型, 'up' 表示上采样, 'down' 表示下采样。
            k_encoder (int): 内容编码器中的卷积核大小。论文建议值为 k_reassembly - 2 [cite: 202]。
            k_reassembly (int): 重组核的大小。
        """
        super().__init__()

        # 参数校验
        assert up_down_type in ['up', 'down'], "up_down_type must be 'up' or 'down'"
        assert k_encoder % 2 == 1, "k_encoder must be an odd number"
        assert k_reassembly % 2 == 1, "k_reassembly must be an odd number"

        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.up_down_type = up_down_type
        self.k_encoder = k_encoder
        self.k_reassembly = k_reassembly
        self.C_reassembly = k_reassembly ** 2

        # 根据论文建议，为上采样和下采样设置不同的压缩后通道数 [cite: 341]
        self.C_m = 64 if self.up_down_type == 'up' else 16

        # --- 1. 核预测模块 ---
        # 1.1 通道压缩器
        self.channel_compressor = nn.Conv2d(in_channels, self.C_m, kernel_size=1)

        # 1.2 内容编码器
        if self.up_down_type == 'up':
            # 上采样时，输出通道为 scale^2 * k_r^2, 后续通过 PixelShuffle 重塑 [cite: 197, 198]
            self.content_encoder = nn.Conv2d(
                self.C_m,
                self.scale_factor ** 2 * self.C_reassembly,
                kernel_size=k_encoder,
                padding=k_encoder // 2,
                stride=1
            )
            # 用于重塑核的空间维度
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        else:  # 'down'
            # 下采样时，直接通过 stride 实现空间尺寸缩小 [cite: 195]
            self.content_encoder = nn.Conv2d(
                self.C_m,
                self.C_reassembly,
                kernel_size=k_encoder,
                padding=k_encoder // 2,
                stride=scale_factor
            )
            self.pixel_shuffle = None

        # 1.3 核归一化器 (在 forward 中通过 F.softmax 实现)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        参数:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。
        返回:
            torch.Tensor: 重组后的特征图。
        """
        # --- 1. 核预测 ---
        # (B, C_m, H, W)
        compressed_x = self.channel_compressor(x)

        # (B, scale^2 * k_r^2, H, W) for upsampling
        # (B, k_r^2, H_out, W_out) for downsampling
        kernels = self.content_encoder(compressed_x)

        # 对上采样生成的核进行重塑
        if self.up_down_type == 'up':
            # (B, k_r^2, H_out, W_out)
            kernels = self.pixel_shuffle(kernels)

        # 核归一化 (空间 softmax)
        # (B, k_r^2, H_out, W_out)
        normalized_kernels = F.softmax(kernels, dim=1)

        # --- 2. 内容感知重组 ---
        if self.up_down_type == 'up':
            return self._reassemble_up(x, normalized_kernels)
        else:  # 'down'
            return self._reassemble_down(x, normalized_kernels)

    def _reassemble_up(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """ 内存高效的上采样重组 """
        N, C, H, W = x.size()

        # 1. 准备重组核 (kernels)
        # 将 (N, k*k, H_out, W_out) 的核变换为 (N, H, W, k*k, up*up)
        # 这里的 up*up 代表每个输入点对应的输出邻域
        k_unfolded = kernels.unfold(2, self.scale_factor, step=self.scale_factor)
        k_unfolded = k_unfolded.unfold(3, self.scale_factor, step=self.scale_factor)
        # (N, k*k, H, W, up, up)
        k_unfolded = k_unfolded.reshape(N, self.C_reassembly, H, W, self.scale_factor ** 2)
        # (N, H, W, k*k, up*up)
        k_unfolded = k_unfolded.permute(0, 2, 3, 1, 4)

        # 2. 准备输入特征邻域 (x)
        # 使用 pad 和 unfold 提取每个点的邻域
        x_padded = F.pad(x, pad=(self.k_reassembly // 2, self.k_reassembly // 2,
                                 self.k_reassembly // 2, self.k_reassembly // 2),
                         mode='constant', value=0)
        x_unfolded = x_padded.unfold(2, self.k_reassembly, step=1)
        x_unfolded = x_unfolded.unfold(3, self.k_reassembly, step=1)
        # (N, C, H, W, k, k)
        x_unfolded = x_unfolded.reshape(N, C, H, W, -1)
        # (N, H, W, C, k*k)
        x_unfolded = x_unfolded.permute(0, 2, 3, 1, 4)

        # 3. 执行矩阵乘法进行加权重组
        # (N, H, W, C, k*k) @ (N, H, W, k*k, up*up) -> (N, H, W, C, up*up)
        out_tensor = torch.matmul(x_unfolded, k_unfolded)
        # (N, H, W, C*up*up)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        # (N, C*up*up, H, W)
        out_tensor = out_tensor.permute(0, 3, 1, 2)

        # 4. 使用 PixelShuffle 将结果重塑为最终输出尺寸
        # (N, C, H*up, W*up)
        out_tensor = F.pixel_shuffle(out_tensor, self.scale_factor)
        return out_tensor

    def _reassemble_down(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """ 下采样重组 """
        B, C, H, W = x.shape
        _, _, H_out, W_out = kernels.shape

        # 使用带步长的 unfold 直接提取下采样位置对应的邻域块
        # (B, C * k_r^2, H_out * W_out)
        unfolded_x = F.unfold(x,
                              kernel_size=self.k_reassembly,
                              padding=self.k_reassembly // 2,
                              stride=self.scale_factor)

        # 调整形状以进行乘法
        # (B, C, k_r^2, L) where L = H_out * W_out
        unfolded_x = unfolded_x.view(B, C, self.C_reassembly, H_out * W_out)
        # (B, 1, k_r^2, L)
        kernels = kernels.reshape(B, 1, self.C_reassembly, H_out * W_out)

        # 执行加权求和
        # (B, C, L)
        output_flat = (unfolded_x * kernels).sum(dim=2)

        # 重塑回图像格式
        # (B, C, H_out, W_out)
        output = output_flat.view(B, C, H_out, W_out)
        return output