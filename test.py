import time
import torch

from ultralytics.nn.new_modules import *

def check(module, args, log=True):
    start_time = time.perf_counter()
    result = module(args).shape
    total_time = time.perf_counter() - start_time
    if log:
        print(f"{type(module)}: {total_time:.4f}s, result:{result}")
    return total_time, result

def meantime(module, args, repeat=20, log=True):
    module(args) #编译一下，忽略编译时间
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(repeat):
        module(args)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    if log:
        print(f"{type(module)}: {total_time/repeat:.4f}s")
    return total_time/repeat

if __name__ == '__main__':
    #N(batch size), C(channels), H(height), W(width)
    in_channels = 64
    x = torch.rand(2, in_channels, 512, 512).to('cuda')
    carafe = CARAFE(in_channels, in_channels).to('cuda')
    dlu = DLUPack(in_channels, 2).to('cuda')
    carafepp = CARAFEplusplus(in_channels, 2).to('cuda')
    dysample = DySample(in_channels).to('cuda')


    meantime(torch.nn.Upsample(scale_factor=2), x)
    meantime(carafe, x)
    meantime(dlu, x)
    meantime(carafepp, x)
    meantime(dysample, x)
