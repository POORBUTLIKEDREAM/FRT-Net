import torch
import torch.nn as nn


class MCFFM(nn.Module):
    """多尺度特征融合模块 (Multi-Channel Feature Fusion Module)"""

    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(in_channels // 2),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(in_channels // 2),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv1d((in_channels // 2) * 3, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(out)