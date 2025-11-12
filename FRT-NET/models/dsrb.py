import torch
import torch.nn as nn
import torch.nn.functional as F


class DSRB(nn.Module):
    """
    Depthwise Separable Residual Block (DSRB)
    Combines depthwise separable convolution, channel attention, and residual connections.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        # Channel Attention (Squeeze-and-Excitation style)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, max(1, out_channels // 8), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max(1, out_channels // 8), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Residual connection (downsample if needed)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        attention = self.channel_attention(out)
        out = out * attention
        out += residual
        return F.gelu(out)