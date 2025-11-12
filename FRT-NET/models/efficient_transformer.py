import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearAttention(nn.Module):
    """
    Efficient Linear Attention Module.
    Reduces complexity by projecting K and V's sequence dimension.
    """

    def __init__(self, d_model, nhead, seq_len, k=64, dropout=0.1, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.k = k
        self.batch_first = batch_first

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Project sequence dim from seq_len (e.g., 32) down to k (e.g., 32)
        self.proj_k = nn.Linear(seq_len, k)
        self.proj_v = nn.Linear(seq_len, k)

        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        if not self.batch_first:
            query, key, value = [x.permute(1, 0, 2) for x in (query, key, value)]

        batch_size, seq_len, _ = query.shape

        q = self.to_q(query).view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(key).view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(value).view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        # Permute to apply linear layer on sequence dim (S)
        # (B, H, S, D_h) -> (B, H, D_h, S)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Apply projection on the last dim (S)
        k_proj = self.proj_k(k)
        v_proj = self.proj_v(v)

        # Permute back: (B, H, D_h, k) -> (B, H, k, D_h)
        k_proj = k_proj.permute(0, 1, 3, 2)
        v_proj = v_proj.permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v_proj)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.to_out(context)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output, attn_probs


class EfficientTransformerBlock(nn.Module):
    """
    Transformer Encoder block using LinearAttention (pre-norm).
    """

    def __init__(self, d_model, nhead, seq_len, dim_feedforward, dropout, k):
        super().__init__()
        self.self_attn = LinearAttention(
            d_model, nhead, seq_len=seq_len, k=k, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout(src2)

        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout(src2)
        return src