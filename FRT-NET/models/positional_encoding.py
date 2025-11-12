import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer ensures 'pe' is part of the model state
        # but not updated by the optimizer.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input x: (batch_size, seq_len, embed_dim)
        """
        # Add positional encoding to the input tensor
        return x + self.pe[:, :x.size(1), :]