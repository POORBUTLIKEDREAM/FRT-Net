import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .dsrb import DSRB
from .dsra import DSRA
from .positional_encoding import PositionalEncoding
from .efficient_transformer import EfficientTransformerBlock


class FRT_Net(nn.Module):
    """
    Fault Diagnosis Model (FRT-Net)
    Combines a CNN backbone (DSRB, DSRA) with an Efficient Transformer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.residual_blocks = nn.Sequential(
            DSRB(64, 64),
            DSRB(64, 128, stride=2),
            DSRB(128, 128),
            DSRB(128, embed_dim, stride=2),
            DSRB(embed_dim, embed_dim)
        )

        self.multiscale_fusion = DSRA(embed_dim)

        # Calculate Transformer input sequence length
        transformer_input_seq_len = self._get_transformer_input_len(config.signal_length)

        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=256)
        self.transformer_blocks = nn.Sequential(
            *[EfficientTransformerBlock(
                d_model=embed_dim,
                nhead=config.num_heads,
                seq_len=transformer_input_seq_len,  # Pass the calculated length (32)
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                k=config.transformer_k
            ) for _ in range(config.num_transformer_layers)]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(config.classifier_dropout_1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(config.classifier_dropout_2),
            nn.Linear(256, config.num_classes)
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(embed_dim, config.num_classes),
            nn.Dropout(config.aux_dropout)
        )

    def _get_transformer_input_len(self, signal_length):
        """Helper function to calculate the output sequence length from the CNN backbone."""

        def get_conv_output_len(L_in, padding, kernel_size, stride):
            return math.floor((L_in + 2 * padding - kernel_size) / stride + 1)

        def get_pool_output_len(L_in, kernel_size, stride):
            return math.floor((L_in - kernel_size) / stride + 1)

        seq_len = signal_length // 2 + 1  # 513
        seq_len = get_conv_output_len(seq_len, padding=3, kernel_size=7, stride=2)  # 257
        seq_len = get_pool_output_len(seq_len, kernel_size=2, stride=2)  # 128

        # DSRB stack
        seq_len = get_conv_output_len(seq_len, padding=1, kernel_size=3, stride=1)  # 128 (DSRB 1)
        seq_len = get_conv_output_len(seq_len, padding=1, kernel_size=3, stride=2)  # 64  (DSRB 2)
        seq_len = get_conv_output_len(seq_len, padding=1, kernel_size=3, stride=1)  # 64  (DSRB 3)
        seq_len = get_conv_output_len(seq_len, padding=1, kernel_size=3, stride=2)  # 32  (DSRB 4)
        seq_len = get_conv_output_len(seq_len, padding=1, kernel_size=3, stride=1)  # 32  (DSRB 5)

        return seq_len  # Final length is 32

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        x = torch.abs(x_fft)

        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.multiscale_fusion(x)

        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_blocks(x)

        x_pooled = self.global_avg_pool(x.permute(0, 2, 1))
        x_pooled = x_pooled.squeeze(-1)
        main_out = self.classifier(x_pooled)

        aux_in = x.mean(dim=1)
        aux_out = self.aux_classifier(aux_in)

        return main_out + self.config.aux_loss_weight * aux_out