import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import CBAM


class CARAFE(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size  # 3
        self.up_factor = up_factor  # 2
        self.down = nn.Conv2d(c1, c1 // 4, 1)  # 1024 256
        self.encoder = nn.Conv2d(c1 // 4,
                                 self.up_factor ** 2 * self.kernel_size ** 2,  # 指数优先级最高
                                 self.kernel_size, 1, self.kernel_size // 2)  # 256 36
        self.out = nn.Conv2d(c1, c2, 1)  # 1024 1024

    def forward(self, x):
        N, C, H, W = x.size()  # 8 1024 20 20
        # 内核预测模块
        kernel_tensor = self.down(x)  # 8 256 20 20
        kernel_tensor = self.encoder(kernel_tensor)  # 8 36 20 20
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # 8 9 40 40
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # 8 9 40 40
        # 滑动窗口，只卷不积
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # 8 9 20 40 2
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # 8 9 20 20 2 2
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2)  # 8 9 20 20 4
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # 8 20 20 9 4

        # 内容感知重组模块
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # 8 1024 22 22
        x = x.unfold(2, self.kernel_size, step=1)  # 8 1024 20 22 3
        x = x.unfold(3, self.kernel_size, step=1)  # 8 1024 20 20 3 3
        x = x.reshape(N, C, H, W, -1)  # 8 1024 20 20 9
        x = x.permute(0, 2, 3, 1, 4)  # 8 20 20 1024 9
        out_tensor = torch.matmul(x, kernel_tensor)  # 8 20 20 1024 4
        out_tensor = out_tensor.reshape(N, H, W, -1)  # 8 20 20 4096
        out_tensor = out_tensor.permute(0, 3, 1, 2)  # 8 4096 20 20
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)  # 8 1024 40 40
        out_tensor = self.out(out_tensor)  # 8 1024 40 40
        # print("up shape:",out_tensor.shape)
        return out_tensor

class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        # self.cbam = CBAM(c1=places * self.expansion, c2=places * self.expansion, )
        self.cbam = CBAM(c1=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
