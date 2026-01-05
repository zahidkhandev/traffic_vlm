# Task 10: Vision Self-Attention

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ModelConfig


class SigLipVisionAttention(nn.Module):
    """
    Multi-head self-attention mechanism for the vision encoder.

    This allows each image patch to attend to every other patch in the sequence.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the attention layer.

        Args:
            config (ModelConfig): Configuration object containing hidden_dim and num_heads.
        """
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.

        Args:
            hidden_states (torch.Tensor): Input states of shape [Batch, Seq_Len, Hidden_Dim].

        Returns:
            torch.Tensor: Attended output states of shape [Batch, Seq_Len, Hidden_Dim].
        """
        batch_size, seq_len, _ = hidden_states.size()

        query = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output
