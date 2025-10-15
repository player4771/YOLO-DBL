import torch

from ultralytics.nn.new_modules.DLU import DLUPack

if __name__ == '__main__':
    x = torch.rand(2, 64, 4, 7).to('cuda')
    dys = DLUPack(64, 2).to('cuda')
    print(f"{x.shape}\n{dys(x).shape}")
