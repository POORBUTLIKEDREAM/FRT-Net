# fault_diagnosis_project/models/FRT_Net.py
import torch.nn as nn
from .components.FFT_Module import FFT_Module
from .components.DSRB import DSRB
from .components.MCFFM import MCFFM
from .components.PositionalEncoding import PositionalEncoding
# Corrected import statement below
from .components.Dual_Head import DualHead


class FRTNet(nn.Module):
    """
    Fault Diagnosis Residual Transformer Network (FRT-Net).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        m_config = config.model

        self.fft = FFT_Module()

        # Initial Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, m_config.initial_conv_out_channels,
                      kernel_size=m_config.initial_conv_kernel_size,
                      stride=m_config.initial_conv_stride,
                      padding=(m_config.initial_conv_kernel_size - 1) // 2),
            nn.BatchNorm1d(m_config.initial_conv_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=m_config.initial_pool_size)
        )

        # Residual Feature Extractor
        res_layers = []
        in_channels = m_config.initial_conv_out_channels
        for out_channels in m_config.residual_channels:
            res_layers.append(DSRB(in_channels, out_channels, stride=2))
            res_layers.append(DSRB(out_channels, out_channels, stride=1))
            in_channels = out_channels
        self.residual_blocks = nn.Sequential(*res_layers)

        self.fusion = MCFFM(m_config.d_model)
        self.pos_encoder = PositionalEncoding(m_config.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m_config.d_model,
            nhead=m_config.n_head,
            dim_feedforward=m_config.dim_feedforward,
            dropout=m_config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=m_config.transformer_layers)

        # Corrected class instantiation below
        self.dual_head = DualHead(config)

    def forward(self, x):
        x = self.fft(x)
        x = self.stem(x)
        x = self.residual_blocks(x)
        x = self.fusion(x)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C) for Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        main_out, aux_out = self.dual_head(x)

        # Combine outputs for training loss, return only main for inference
        if not self.training:
            return main_out
        return main_out + self.config.model.aux_head_weight * aux_out
