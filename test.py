import torch
torch.set_float32_matmul_precision('high')

from ultralytics.nn.modules import *
from ultralytics.nn.modules_upsample import *
from ultralytics.nn.modules_attention import *

from global_utils import check, label_image_tea

def upsample_test(in_channels:int=64, size:int=64):
    # N(batch size), C(channels), H(height), W(width)
    x = torch.rand(2, in_channels, size, size).to('cuda')
    y = torch.rand(2, in_channels, size * 2, size * 2).to('cuda')

    modules = {
        torch.nn.Upsample(scale_factor=2, mode='bilinear'): x,
        CARAFE(in_channels, in_channels): x,
        DLUPack(in_channels): x,
        CARAFEplusplus(in_channels): x,
        DySample(in_channels): x,
        EUCB(in_channels, in_channels): x,
        MEUM(in_channels): x,
        CARAFEPack(in_channels): x,
        SAPA(in_channels): (y, x),
        FGA(64, upscale=2): None,
        SFAU(in_channels, in_channels, scale=2): None,
        torch.compile(AnyUpPack(in_channels)): None,
        torch.compile(FADE_H2L(in_channels, in_channels)): None,
        torch.compile(FADE_L2H(in_channels, in_channels)): None,
        torch.compile(FADELite(in_channels, in_channels)): None,
        torch.compile(LDA_AQU(in_channels)): x,
        CARAFE_XiaLiPKU(in_channels): None,
    }

    for module, input in modules.items():
        if input is not None:
            if not isinstance(input, tuple):
                check(module.to('cuda'), input)
            else:
                check(module.to('cuda'), *input)


def attention_test(in_channels:int=64, size:int=256):
    x_nchw = torch.rand(4, in_channels, size, size).to('cuda')
    x_nhwc = x_nchw.permute(0, 2, 3, 1).to('cuda')

    modules = {# <module: input>, None代表可以运行，但会爆显存
        CBAM(in_channels): x_nchw,
        biformer(in_channels=in_channels, model_size='tiny'): x_nchw,
        biformer(in_channels=in_channels, model_size='small'): x_nchw,
        biformer(in_channels=in_channels, model_size='base'): x_nchw,
        DAT(in_chans=in_channels): None,
        BAM(in_channels): x_nchw,
        CoordAttention(in_channels, in_channels): x_nchw,
        CPCA(in_channels, in_channels): x_nchw,
        DeBiAttention(in_channels, n_win=8): None,
        ECALayer(in_channels): x_nchw,
        EfficientAttention(in_channels): x_nchw,
        EMA(in_channels): x_nchw,
        FullyAttentionalBlock(in_channels): x_nchw,
        HiLo(in_channels): None,
        NonLocalBlock2D(in_channels): None,
        SELayer(in_channels): x_nchw,
        SimAM(in_channels): x_nchw,
        volo_d1(img_size=size, in_chans=in_channels): x_nchw,
        volo_d2(img_size=size, in_chans=in_channels): x_nchw,
        volo_d3(img_size=size, in_chans=in_channels): x_nchw,
        volo_d4(img_size=size, in_chans=in_channels): x_nchw,
        volo_d5(img_size=size, in_chans=in_channels): None, #过慢. 耗时大约是d4的30倍, d1的150倍
        SLA(in_channels): x_nchw,
        GAM(in_channels, in_channels): x_nchw,
        SwinTransformer(in_channels, in_channels): x_nchw,
        MLCA(in_channels): x_nchw,
        ELA(in_channels): x_nchw,
    }

    for module, input in modules.items():
        if input is not None:
            check(module.to('cuda'), input)

if __name__ == '__main__':
    img_file = r"E:\Projects\Datasets\tea_leaf_diseases_v4\train\images\IMG_20230612_151845_jpg.rf.f66a145758c7c6dd4c6ac816c813601a.jpg"
    #label_image_tea(img_file)
    attention_test()