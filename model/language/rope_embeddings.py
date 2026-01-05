# Task 13
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    Fixed for Pylance strict type checking.
    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len is None:
            current_seq_len = x.shape[2]
        else:
            current_seq_len = seq_len

        if current_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=current_seq_len, device=x.device)

        return (
            self.cos_cached[:, :, :current_seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :current_seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to Query and Key states."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
