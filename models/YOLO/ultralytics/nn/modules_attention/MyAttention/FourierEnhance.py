import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEnhance(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.channels = channels

        self.freq_mask = None #频域遮罩，用于频域增强，由于需要输入尺寸所以这里没有初始化
        self.delta_phase = nn.Parameter(torch.zeros(1)) #相位

    def forward(self, x):
        N, C, H, W = x.shape
        pad_W = torch.pow(2, torch.ceil(torch.log2(W))).type(torch.int64) - W
        pad_H = torch.pow(2, torch.ceil(torch.log2(H))).type(torch.int64) - H
        x = F.pad(x, (0, pad_W, 0, pad_H), mode='constant', value=0) #填充到2的次方尺寸，以使用cuFFT

        x_f = torch.fft.fft2(x)
        x_f = torch.fft.fftshift(x_f)

        #相位调整
        magnitude = torch.abs(x_f)
        phase = torch.angle(x_f) + self.delta_phase
        x_f = magnitude * torch.exp(1j * phase)

        #遮罩
        self.freq_mask = nn.Parameter(torch.ones_like(x_f))
        x_f = x_f * self.freq_mask

        x_f = torch.fft.ifftshift(x_f)
        x = torch.fft.ifft2(x_f).real
        x = x[:N, :C, :H, :W]
        return x

if __name__ == '__main__':
    import joblib
    from global_utils import plot_feature_maps

    layer_indexes = (26, 31, 36)
    results = joblib.load(rf"E:\Projects\PyCharm\Paper2\global_utils\cache\{hash(layer_indexes)}.cache")

    input = results[layer_indexes[0]]['output']
    channels = input.shape[-3]

    model = FourierEnhance(channels=channels)
    with torch.inference_mode():
        output = model.to('cuda').eval()(input.to('cuda'))

    plot_feature_maps(input, output)