#https://github.com/myownskyW7/CARAFE/blob/master/carafe/carafe.py
#carafe_ext和carafe_naive_ext 由Gemini根据C++源码改编

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.module import Module


class CarafeNaiveExt:

    @staticmethod
    def forward(features, masks, kernel_size, group_size, scale_factor, output):
        n, c, h, w = features.size()
        c_g = c // group_size
        k = kernel_size
        s = scale_factor

        # 1. 特征准备: 提取每个输入像素的 k*k 邻域特征块
        # (n, c, h, w) -> (n, g, c_g, h, w, k*k)
        features_unfolded = F.unfold(features, kernel_size=k, padding=(k - 1) // 2)
        feature_patches = features_unfolded.view(n, group_size, c_g, k * k, h, w).permute(0, 1, 2, 4, 5, 3)

        # 2. 上采样核准备: 将 mask 重组为与特征块对应的 s*s 上采样核
        # (n, g*k*k, h*s, w*s) -> (n, g, k*k, h, w, s*s)
        upsample_kernels = masks.view(n, group_size, k * k, h, s, w, s).permute(0, 1, 3, 5, 2, 4, 6)
        upsample_kernels = upsample_kernels.reshape(n, group_size, h, w, k * k, s * s)

        # 3. 核心计算: einsum 高效执行加权求和
        # 特征块: (n, g, c_g, h, w, k*k), 上采样核: (n, g, h, w, k*k, s*s) -> 输出块: (n, g, c_g, h, w, s*s)
        output_patches = torch.einsum('ngchwi,nghwis->ngchws', feature_patches, upsample_kernels)

        # 4. 输出重组: 将输出块通过 pixel_shuffle 还原为目标形状
        # (n, g, c_g, h, w, s*s) -> (n, c, h_out, w_out)
        output_reshaped = output_patches.permute(0, 1, 2, 5, 3, 4).contiguous().view(n, c * (s * s), h, w)
        output.copy_(F.pixel_shuffle(output_reshaped, s))

        return output

    @staticmethod
    def backward(grad_output, features, masks, kernel_size, group_size,
                 scale_factor, grad_input, grad_masks):
        n, c, h, w = features.size()
        c_g = c // group_size
        k = kernel_size
        s = scale_factor
        g_k_k = group_size * k * k

        # 1. 梯度准备: 将 grad_output 逆操作, 得到每个输出块的梯度
        # (n, c, h*s, w*s) -> (n, g, c_g, h, w, s*s)
        grad_output_reshaped = grad_output.view(n, c, h, s, w, s).permute(0, 1, 3, 5, 2, 4)
        grad_output_patches = grad_output_reshaped.reshape(n, group_size, c_g, s * s, h, w).permute(0, 1, 2, 4, 5, 3)

        # 2. 特征准备 (同 forward)
        features_unfolded = F.unfold(features, kernel_size=k, padding=(k - 1) // 2)
        feature_patches = features_unfolded.view(n, group_size, c_g, k * k, h, w).permute(0, 1, 2, 4, 5, 3)

        # 3. 计算 grad_masks
        # grad_output_patches: (n,g,c,h,w,s), feature_patches: (n,g,c,h,w,i) -> grad_kernels: (n,g,h,w,i,s)
        grad_kernels = torch.einsum('ngchws,ngchwi->nghwis', grad_output_patches, feature_patches)
        grad_kernels = grad_kernels.permute(0, 1, 4, 2, 5, 3).contiguous().view(n, g_k_k, h * s, w * s)
        grad_masks.copy_(grad_kernels)

        # 4. 计算 grad_input
        # 上采样核准备 (同 forward)
        upsample_kernels = masks.view(n, group_size, k * k, h, s, w, s).permute(0, 1, 3, 5, 2, 4, 6)
        upsample_kernels = upsample_kernels.reshape(n, group_size, h, w, k * k, s * s)
        # grad_output_patches: (n,g,c,h,w,s), kernels: (n,g,h,w,i,s) -> grad_feature_patches: (n,g,c,h,w,i)
        grad_feature_patches = torch.einsum('ngchws,nghwis->ngchwi', grad_output_patches, upsample_kernels)
        grad_feature_patches = grad_feature_patches.permute(0, 1, 2, 5, 3, 4).contiguous().view(n, c * k * k, h * w)
        grad_input.copy_(F.fold(grad_feature_patches, (h, w), kernel_size=k, padding=(k - 1) // 2))


class CarafeExt:

    @staticmethod
    def forward(features, rfeatures, masks, rmasks, kernel_size, group_size,
                scale_factor, routput, output):
        return CarafeNaiveExt.forward(features, masks, kernel_size, group_size,
                                      scale_factor, output)

    @staticmethod
    def backward(grad_output, rfeatures, masks, kernel_size, group_size,
                 scale_factor, rgrad_output, rgrad_input_hs, rgrad_input,
                 rgrad_masks, grad_input, grad_masks):
        return CarafeNaiveExt.backward(grad_output, rfeatures, masks,
                                       kernel_size, group_size, scale_factor,
                                       grad_input, grad_masks)

carafe_ext = CarafeExt()
carafe_naive_ext = CarafeNaiveExt()


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class CARAFENaiveFunction(Function):

    @staticmethod
    def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        if features.is_cuda:
            carafe_naive_ext.forward(features, masks, kernel_size, group_size,
                                     scale_factor, output)
        else:
            raise NotImplementedError

        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        features, masks = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        grad_input = torch.zeros_like(features)
        grad_masks = torch.zeros_like(masks)
        carafe_naive_ext.backward(grad_output.contiguous(), features, masks,
                                  kernel_size, group_size, scale_factor,
                                  grad_input, grad_masks)

        return grad_input, grad_masks, None, None, None


carafe_naive = CARAFENaiveFunction.apply


class CARAFENaive(Module):

    def __init__(self, kernel_size, group_size, scale_factor):
        super(CARAFENaive, self).__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features, masks):
        return CARAFENaiveFunction.apply(features, masks, self.kernel_size,
                                         self.group_size, self.scale_factor)


class CARAFEFunction(Function):

    @staticmethod
    def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        routput = features.new_zeros(output.size(), requires_grad=False)
        rfeatures = features.new_zeros(features.size(), requires_grad=False)
        rmasks = masks.new_zeros(masks.size(), requires_grad=False)
        if features.is_cuda:
            carafe_ext.forward(features, rfeatures, masks, rmasks, kernel_size,
                               group_size, scale_factor, routput, output)
        else:
            raise NotImplementedError

        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks, rfeatures)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        features, masks, rfeatures = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_masks = torch.zeros_like(masks, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_masks = torch.zeros_like(masks, requires_grad=False)
        carafe_ext.backward(grad_output.contiguous(), rfeatures, masks,
                            kernel_size, group_size, scale_factor,
                            rgrad_output, rgrad_input_hs, rgrad_input,
                            rgrad_masks, grad_input, grad_masks)
        return grad_input, grad_masks, None, None, None, None


carafe = CARAFEFunction.apply


class CARAFE(Module):
    """ CARAFE: Content-Aware ReAssembly of FEatures

    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        kernel_size (int): reassemble kernel size
        group_size (int): reassemble group size
        scale_factor (int): upsample ratio

    Returns:
        upsampled feature map
    """

    def __init__(self, kernel_size, group_size, scale_factor):
        super(CARAFE, self).__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features, masks):
        return CARAFEFunction.apply(features, masks, self.kernel_size,
                                    self.group_size, self.scale_factor)


class CARAFEPack(nn.Module):
    """ A unified package of CARAFE upsampler that contains:
    1) channel compressor 2) content encoder 3) CARAFE op

    Official implementation of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

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
        super(CARAFEPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels,
                                            1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / (self.up_kernel * self.up_kernel))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x

if __name__ == '__main__':
    x = torch.rand(2, 64, 512, 512).to('cuda')
    carafe_pack = CARAFEPack(64, 2).to('cuda')
    print(carafe_pack(x).shape)