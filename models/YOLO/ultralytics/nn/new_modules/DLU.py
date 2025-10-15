import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import xavier_init, normal_init

def carafe(features, masks, kernel_size, group_size, scale_factor):
    """
    CARAFE 算子的纯 PyTorch 实现。如有可能可以编译为CUDA算子

    Args:
        features (Tensor): 输入特征图, shape (N, C, H, W)
        masks (Tensor): 预测的上采样核, shape (N, G*k*k, H_out, W_out)
        kernel_size (int): 重组核的大小 (k_up)
        group_size (int): 分组数 (G)
        scale_factor (int): 上采样倍率 (σ)

    Returns:
        Tensor: 上采样后的特征图
    """
    N, C, H, W = features.shape
    H_out, W_out = H * scale_factor, W * scale_factor

    if C % group_size != 0:
        raise ValueError(f'输入通道数 {C} 必须能被分组数 {group_size} 整除')

    C_g = C // group_size  # 每个组的通道数

    # 使用 F.unfold 高效地提取滑动的局部块 (patches)
    unfolded_features = F.unfold(
        features,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    unfolded_features = unfolded_features.view(
        N, group_size, C_g, kernel_size * kernel_size, H, W
    )

    # 为输出图的每个位置找到对应的输入块
    h_idx = torch.arange(H_out, device=features.device) // scale_factor
    w_idx = torch.arange(W_out, device=features.device) // scale_factor

    selected_features = unfolded_features[:, :, :, :, h_idx, :][:, :, :, :, :, w_idx]

    # 特征重组 (内积操作)
    masks = masks.view(N, group_size, kernel_size * kernel_size, H_out, W_out).unsqueeze(2)

    reassembled_features = selected_features * masks
    output = torch.sum(reassembled_features, dim=3)

    return output.view(N, C, H_out, W_out)


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
                 scale_factor,
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
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        offset = self.conv_offset(compressed_x)
        mask = self.kernel_space_generator(compressed_x)
        mask = self.kernel_space_normalizer(mask)

        mask_ = self.kernel_space_expander(offset, mask)

        x = self.feature_reassemble(x, mask_)
        return x