import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import xavier_init, normal_init


def carafe(features, masks, kernel_size, up_factor):
    """
    使用 unfold + matmul 的逻辑来实现特征重组。
    Args:
        features (Tensor): 低分辨率输入特征图, (N, C, H, W)
        masks (Tensor): 预测的上采样核, (N, k*k, H_out, W_out)
        kernel_size (int): 重组核的大小 (k)
        up_factor (int): 上采样倍率 (σ)

    Returns:
        Tensor: 高分辨率输出特征图
    """
    N, C, H, W = features.size()

    # --- 1. 准备上采样核 (masks) ---
    # 将 masks 变形以适应 matmul
    kernel_tensor = masks.unfold(2, up_factor, step=up_factor)
    kernel_tensor = kernel_tensor.unfold(3, up_factor, step=up_factor)
    kernel_tensor = kernel_tensor.reshape(N, kernel_size ** 2, H, W, up_factor ** 2)
    kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # 最终形状: (N, H, W, k*k, σ*σ)

    # --- 2. 准备输入特征 (features) ---
    x = F.pad(features, pad=(kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
              mode='constant', value=0)
    x = x.unfold(2, kernel_size, step=1)
    x = x.unfold(3, kernel_size, step=1)
    x = x.reshape(N, C, H, W, -1)
    x = x.permute(0, 2, 3, 1, 4)  # 最终形状: (N, H, W, C, k*k)

    # --- 3. 执行特征重组 ---
    # (N, H, W, C, k*k) @ (N, H, W, k*k, σ*σ) -> (N, H, W, C, σ*σ)
    out_tensor = torch.matmul(x, kernel_tensor)

    # --- 4. 将输出重组为图像格式 ---
    out_tensor = out_tensor.reshape(N, H, W, -1)
    out_tensor = out_tensor.permute(0, 3, 1, 2)
    out_tensor = F.pixel_shuffle(out_tensor, up_factor)

    return out_tensor

def carafe2(features, masks, kernel_size, group_size, up_factor):
    """
    将 group_size 参数添加回 unfold + matmul 实现中。
    group_size (分组数) 的作用是将输入特征图的通道 (Channels) 分成若干个独立的组，然后为每一组分别进行内容感知的特征重组。
    """
    N, C, H, W = features.size()

    # 检查通道数是否能被分组整除
    assert C % group_size == 0, f'输入通道数 {C} 必须能被分组数 {group_size} 整除'
    C_g = C // group_size  # 每个组的通道数

    # --- 1. 准备上采样核 (masks) ---
    # masks 的输入形状假定为 (N, G * k*k, H_out, W_out), 将其 reshape 以便后续处理
    masks = masks.view(N, group_size, kernel_size ** 2, H * up_factor, W * up_factor)

    kernel_tensor = masks.unfold(3, up_factor, step=up_factor)  # 在 H_out 维度上 unfold
    kernel_tensor = kernel_tensor.unfold(4, up_factor, step=up_factor)  # 在 W_out 维度上 unfold
    # -> (N, G, k*k, H, W, σ, σ)
    kernel_tensor = kernel_tensor.reshape(N, group_size, kernel_size ** 2, H, W, up_factor ** 2)
    # -> (N, G, H, W, k*k, σ*σ)
    kernel_tensor = kernel_tensor.permute(0, 1, 3, 4, 2, 5)

    # --- 2. 准备输入特征 (features) ---
    features = features.view(N, group_size, C_g, H, W)# 将 features 分组

    x = F.pad(features, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'constant', 0)
    x = x.unfold(3, kernel_size, step=1)  # 在 H 维度上 unfold
    x = x.unfold(4, kernel_size, step=1)  # 在 W 维度上 unfold
    # -> (N, G, Cg, H, W, k, k)
    x = x.reshape(N, group_size, C_g, H, W, -1)
    # -> (N, G, H, W, Cg, k*k)
    x = x.permute(0, 1, 3, 4, 2, 5)

    # --- 3. 执行特征重组 ---
    # (N, G, H, W, Cg, k*k) @ (N, G, H, W, k*k, σ*σ) -> (N, G, H, W, Cg, σ*σ)
    out_tensor = torch.matmul(x, kernel_tensor)

    # --- 4. 将输出重组为图像格式 ---
    # (N, G, H, W, Cg, σ*σ) -> (N, G, Cg, σ*σ, H, W) -> (N, C, H, W, σ*σ) -> (N, C, H, up_factor, W, up_factor)
    out_tensor = out_tensor.permute(0, 1, 4, 2, 3, 5).reshape(N, C*up_factor**2, H, W)
    out_tensor = F.pixel_shuffle(out_tensor, up_factor)

    return out_tensor

class DLUPack(nn.Module):
    """
    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels,
                 scale_factor=2,
                 up_kernel=5,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64):
        super(DLUPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        # k_up. 对于需要更大感受野的任务可以增大以提高精度，但计算成本也会增加
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        # 64时精度最高，降低该值(如32)可以显著减少后续卷积层的计算量和参数量从而提升性能，但会带来轻微的精度损失
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels, 1)
        self.kernel_space_generator = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.conv_offset = nn.Conv2d(
            self.compressed_channels,
            self.up_group * 2 * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.kernel_space_generator, std=0.001)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()


    def kernel_space_normalizer(self, mask):
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, mask_c, h, w).contiguous()
        return mask

    def kernel_space_expander(self, offset, mask):
        device = offset.device
        n, _, h, w = offset.size()
        offset = F.pixel_shuffle(offset, self.scale_factor)
        offset = offset.permute(0,2,3,1)
        offset[:,:,:,0] = offset[:,:,:,0] * 1/(w-1)*2
        offset[:,:,:,1] = offset[:,:,:,1] * 1/(h-1)*2

        new_h = torch.repeat_interleave(torch.linspace(-1, 1, h, device=device),self.scale_factor)
        new_h = new_h.view(-1, 1).repeat(1, self.scale_factor*w) #一整行太长，拆分了一下
        new_w = torch.repeat_interleave(torch.linspace(-1, 1, w, device=device),self.scale_factor)
        new_w = new_w.repeat(self.scale_factor*h, 1)

        grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        grid_ = grid.expand(n,-1,-1,-1)
        offset = grid_ + offset
        mask_ = F.grid_sample(mask, offset,padding_mode='border',align_corners=True)
        return mask_

    def feature_reassemble(self, x, mask):
        #x = carafe(x, mask, self.up_kernel, self.scale_factor)
        x = carafe2(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        offset = self.conv_offset(compressed_x)
        mask = self.kernel_space_generator(compressed_x)
        mask = self.kernel_space_normalizer(mask)

        mask_ = self.kernel_space_expander(offset, mask)

        x = self.feature_reassemble(x, mask_)
        return x