import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import xavier_init, normal_init


class CARAFE(nn.Module):
    def __init__(self, kernel_size:int, group_size:int, scale_factor:int):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor
        self.padding = (kernel_size - 1) // 2

    def forward(self, features, masks):
        """
        Args:
            features: (N, C, H, W)
            masks: (N, Group * Kernel^2, H_up, W_up)
        """
        N, C, H, W = features.shape
        kernel_size = self.kernel_size
        group_size = self.group_size
        scale = self.scale_factor

        # 1. Unfold: 提取滑动窗口特征
        # Output: (N, C * K*K, H * W)
        unfolded_feat = F.unfold(features, kernel_size=kernel_size, padding=self.padding)
        # Reshape: (N, C, K*K, H, W)
        unfolded_feat = unfolded_feat.view(N, C, kernel_size * kernel_size, H, W)

        # 2. Upsample: 将特征邻域扩展到输出分辨率
        # CARAFE 的定义是：输出点 (x', y') 对应输入点 (x'/s, y'/s) 的邻域
        # 因此，对于每个 s*s 的输出块，它们共享同一个输入中心的邻域特征
        # 使用 repeat_interleave 实现最近邻上采样
        # (N, C, K*K, H, W) -> (N, C, K*K, H*scale, W*scale)
        unfolded_feat = unfolded_feat.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)

        # 3. Apply Mask: 加权求和
        # Mask shape: (N, Group * K^2, H_up, W_up)
        # Feature shape: (N, C, K^2, H_up, W_up)

        if group_size == 1:
            # 常见情况 G=1，Mask 直接广播到所有通道
            # Mask: (N, 1, K^2, H_up, W_up)
            masks = masks.unsqueeze(1)
            # 逐元素相乘并在 K^2 维度求和
            output = (unfolded_feat * masks).sum(dim=2)
        else:
            # G > 1，通道分组处理
            assert C % group_size == 0
            C_per_group = C // group_size

            # Reshape 特征以匹配分组: (N, G, C_sub, K^2, H_up, W_up)
            feat_view = unfolded_feat.view(N, group_size, C_per_group, kernel_size * kernel_size, H * scale, W * scale)
            # Reshape 掩码: (N, G, 1, K^2, H_up, W_up)
            mask_view = masks.view(N, group_size, 1, kernel_size * kernel_size, H * scale, W * scale)

            # 组内加权求和: (N, G, C_sub, H_up, W_up)
            output = (feat_view * mask_view).sum(dim=3)
            # 展平回原始通道: (N, C, H_up, W_up)
            output = output.view(N, C, H * scale, W * scale)

        return output

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
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
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
        self.carafe = CARAFE(self.up_kernel, self.up_group, self.scale_factor)

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
        n, _, h, w = offset.size()
        offset = F.pixel_shuffle(offset, self.scale_factor)
        offset = offset.permute(0,2,3,1)
        offset[:,:,:,0] = offset[:,:,:,0] * 1/(w-1)*2
        offset[:,:,:,1] = offset[:,:,:,1] * 1/(h-1)*2

        new_h = torch.repeat_interleave(torch.linspace(-1, 1, h),self.scale_factor).view(-1, 1).repeat(1, self.scale_factor*w)
        new_w = torch.repeat_interleave(torch.linspace(-1, 1, w),self.scale_factor).repeat(self.scale_factor*h, 1)

        grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        grid_ = grid.expand(n,-1,-1,-1)
        grid_ = grid_.to(offset)
        offset = grid_ + offset
        mask_ = F.grid_sample(mask, offset,padding_mode='border',align_corners=True)
        return mask_

    def feature_reassemble(self, x, mask):
        x = self.carafe(x, mask)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        offset = self.conv_offset(compressed_x)
        mask = self.kernel_space_generator(compressed_x)
        mask = self.kernel_space_normalizer(mask)

        mask_ = self.kernel_space_expander(offset, mask)

        x = self.feature_reassemble(x, mask_)
        return x