import torch
import torch.nn as nn
import torch.nn.functional as F


class DSRA(nn.Module):
    """
    Depthwise Separable Residual Aggregation (DSRA)
    A multi-scale feature fusion module.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2

        # Branch 1: Kernel 3
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.GELU()
        )

        # Branch 2: Kernel 5
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.GELU()
        )

        # Branch 3: Kernel 7
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(mid_channels),
            nn.GELU()
        )

        # Fusion layer (1x1 Conv)
        fused_channels = mid_channels * 3
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # Concatenate along the channel dimension
        out = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(out)