#https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv5.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, RmsNorm2d

from .MobileNetv4 import UniversalInvertedBottleneckBlock

class MobileNetV5MultiScaleFusionAdapter(nn.Module):
  """Multi-layer fusion token adapter.

  Args:
    in_chs: list of input channel counts for each feature scale.
    out_chs: The number of output channels.
    output_resolution: The output resolution.
    expansion_ratio: The FFN expansion ratio.
    interpolation_mode: The upsampling interpolation mode.
    layer_scale_init_value: The initial value of the layer scale, no layer scale if None.
  """

  def __init__(
        self,
        in_chs:int|list[int],
        out_chs: int,
        output_resolution: int,
        expansion_ratio: float = 2.0,
        interpolation_mode: str = "nearest",
        layer_scale_init_value:float=None,
        no_skip: bool = True,
        norm_layer=None, #LayerType
        device=None,
        dtype=None,
  ):
    dd = {'device': device, 'dtype': dtype}
    super().__init__()
    self.in_channels = sum(in_chs) if isinstance(in_chs, (list,tuple)) else in_chs
    self.out_channels = out_chs
    self.output_resolution = to_2tuple(output_resolution)
    self.expansion_ratio = expansion_ratio
    self.interpolation_mode = interpolation_mode
    self.layer_scale_init_value = layer_scale_init_value
    self.no_skip = no_skip

    norm_layer = norm_layer or RmsNorm2d
    self.ffn = UniversalInvertedBottleneckBlock(
        inp=self.in_channels,
        oup=self.out_channels,
        start_dw_kernel_size=0, #FFN通常不需要起始DW卷积
        middle_dw_kernel_size=0, #0表示纯FFN(1x1 conv)
        middle_dw_downsample=False, #不进行下采样
        stride=1,
        expand_ratio=self.expansion_ratio,
        use_layer_scale=(self.layer_scale_init_value is not None)
    )

    self.norm = norm_layer(self.out_channels, **dd)

  def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
    # Inputs list of [B, C, H, W] tensors
    high_resolution = inputs[0].shape[-2:]  # Assuming the first input is the highest resolution.
    resized_inputs = []
    for _, img in enumerate(inputs):
        feat_size = img.shape[-2:]
        if feat_size[0] < high_resolution[0] or feat_size[1] < high_resolution[1]:
            img = F.interpolate(img, size=high_resolution, mode=self.interpolation_mode)
        resized_inputs.append(img)

    channel_cat_imgs = torch.cat(resized_inputs, dim=1)  # Cat on channel dim, must equal self.in_channels
    img = self.ffn(channel_cat_imgs)

    if high_resolution[0] != self.output_resolution[0] or high_resolution[1] != self.output_resolution[1]:
        # Interpolate / pool to target output_resolution if highest feature resolution differs
        if (
            high_resolution[0] % self.output_resolution[0] != 0 or
            high_resolution[1] % self.output_resolution[1] != 0
        ):
            img = F.interpolate(img, size=self.output_resolution, mode="bilinear")
        else:
            h_strides = int(high_resolution[0] // self.output_resolution[0])
            w_strides = int(high_resolution[1] // self.output_resolution[1])
            img = F.avg_pool2d(
                img,
                kernel_size=(h_strides, w_strides),
                stride=(h_strides, w_strides),
            )

    img = self.norm(img)

    return img
