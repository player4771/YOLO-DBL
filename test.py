import re
import time
import torch

from ultralytics.nn.modules_upsample import *
from ultralytics.nn.modules_attention import *

def class_name(class_):
    #如: <class 'ultralytics.nn.modules_attention.BiFormer.biformer.BiFormer'> -> BiFormer
    return re.search(r"<class '.*\.(.*)'>", str(type(class_))).group(1)

def check(module, *args, repeat=10, log=True):
    result = module(*args)  # 运行一下，忽略编译时间
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(repeat):
        module(*args)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    if log:
        print(f"{class_name(module)}: {total_time:.4f}s, result:{result}")
    return total_time, result

def meantime(module, *args, repeat=10, log=True, adjust=25):
    print(f"{class_name(module)}:".ljust(adjust), end='')
    total_time, result = check(module, *args, repeat=repeat, log=False)
    if log:
        try:
            print(f"-> {result.shape},".ljust(40), f"{total_time/repeat}s")
        except:
            print("-> [Unknown result],".ljust(40), f"{total_time/repeat}s")
    return total_time/repeat, result

def upsample_test():
    # N(batch size), C(channels), H(height), W(width)
    in_channels = 64
    size = 256
    x = torch.rand(2, in_channels, size, size).to('cuda')
    y = torch.rand(2, in_channels, size * 2, size * 2).to('cuda')

    #meantime(torch.nn.Upsample(scale_factor=2), x)
    meantime(CARAFE(in_channels, in_channels).to('cuda'), x)
    meantime(DLUPack(in_channels).to('cuda'), x)
    #meantime(CARAFEplusplus(in_channels).to('cuda'), x)
    #meantime(DySample(in_channels).to('cuda'), x)
    #meantime(EUCB(in_channels, in_channels).to('cuda'), x)
    #meantime(MEUM(in_channels).to('cuda'), x)
    #meantime(CARAFEPack(in_channels).to('cuda'), x)
    meantime(SAPA(in_channels).to('cuda'), y, x)
    #meantime(FGA(64, out_dim=64, upscale=2).to('cuda'), x)

def attention_test():
    in_channels = 64
    size = 256
    x_nchw = torch.rand(4, in_channels, size, size).to('cuda')
    x_nhwc = x_nchw.permute(0, 2, 3, 1).to('cuda')

    modules = {# <module: input>, None代表可以运行，但会爆显存
        biformer(in_channels=in_channels, model_size='tiny'): x_nchw,
        biformer(in_channels=in_channels, model_size='small'): x_nchw,
        biformer(in_channels=in_channels, model_size='base'): x_nchw,
        BiLevelRoutingAttention(in_channels,n_win=8): None, #x_nhwc,
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
    }

    for module, input in modules.items():
        if input is not None:
            meantime(module.to('cuda'), input)

if __name__ == '__main__':
    attention_test()