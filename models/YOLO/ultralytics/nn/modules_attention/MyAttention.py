import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from global_utils import plot_feature_maps

class LaplacianSharpen(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        kernal = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        kernal = kernal.view(1,1,3,3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernal)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)

def sobel_kernels(device, dtype):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device)/4.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype, device=device)/4.0
    return kx.view(1,1,3,3), ky.view(1,1,3,3)

class EdgeAwareAttention(nn.Module):
    def __init__(self, channels, reduction=16, ksize=7, alpha=None, beta=None):
        super().__init__()
        self.channels = channels
        # 空间注意力：输入4通道 (avgX, maxX, avgG, maxG)
        self.spatial = nn.Conv2d(4, 1, ksize, padding=ksize//2, bias=True)
        # 通道注意力：MLP
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        # 残差增益（非负）
        self.alpha = nn.Parameter(torch.tensor(0.0 if alpha is None else alpha))
        self.beta  = nn.Parameter(torch.tensor(0.0 if beta is None else beta))

        # 注册 Sobel 作为 buffer，随设备/精度移动
        kx, ky = sobel_kernels('cpu', torch.float32)  # 占位后在forward搬
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):  # x: B,C,H,W
        B, C, H, W = x.shape
        # depthwise Sobel（自动广播到通道）
        kx = self.kx.to(device=x.device, dtype=x.dtype).repeat(C,1,1,1)
        ky = self.ky.to(device=x.device, dtype=x.dtype).repeat(C,1,1,1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        g  = torch.sqrt(gx*gx + gy*gy + 1e-12) # 边缘幅值

        # ---- 通道注意力 Ac ----
        c_vec = g.mean(dim=(-1,-2)) # B,C
        w = torch.sigmoid(self.mlp(c_vec)).view(B,C,1,1)

        # ---- 空间注意力 As ----
        avg_x = x.mean(1, keepdim=True)
        max_x = x.amax(1, keepdim=True)
        avg_g = g.mean(1, keepdim=True)
        max_g = g.amax(1, keepdim=True)
        s_in = torch.cat([avg_x, max_x, avg_g, max_g], dim=1)  # B,4,H,W
        s = torch.sigmoid(self.spatial(s_in))

        # 残差式增强（softplus保证非负增益）
        alpha = F.softplus(self.alpha)
        beta  = F.softplus(self.beta)
        y = x * (1 + alpha * s) * (1 + beta * w)
        return y

if __name__ == '__main__':
    layer_indexes = (26, 31, 36)
    results = joblib.load(rf"E:\Projects\PyCharm\Paper2\global_utils\cache\{hash(layer_indexes)}.cache")

    input = results[layer_indexes[0]]['output'].to('cuda')
    channels = input.shape[-3]
    assert isinstance(input, torch.Tensor), TypeError(f'Invalid type: {type(input)}, expected torch.Tensor')

    #model = LaplacianSharpen(channels=channels])
    model = EdgeAwareAttention(channels=channels, ksize=3,alpha=1.2,beta=0.8)
    #print(model.weight.device, input.device)
    with torch.no_grad():
        output = model.to('cuda').eval()(input)

    plot_feature_maps(input, output)