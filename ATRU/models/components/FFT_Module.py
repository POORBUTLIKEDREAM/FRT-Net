import torch
import torch.nn as nn


class FFT_Module(nn.Module):
    """傅里叶变换特征提取模块"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 应用傅里叶变换
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        # 取绝对值获取幅度谱
        x = torch.abs(x_fft)
        return x