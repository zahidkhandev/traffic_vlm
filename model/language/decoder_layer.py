# Task 12: Single Decoder Block
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.language.rope_embeddings import apply_rotary_pos_emb


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class GemmaMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    def __init__(self, config: ModelConfig, is_cross_attention=False):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        output_attentions=False,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        if self.is_cross_attention:
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if not self.is_cross_attention and rotary_emb is not None:
            cos, sin = rotary_emb(value_states, seq_len=seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        )
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights

        return attn_output, None


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.self_attn = GemmaAttention(config, is_cross_attention=False)
        self.input_layernorm = GemmaRMSNorm(self.hidden_dim)
        self.cross_attn = GemmaAttention(config, is_cross_attention=True)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_dim)
        self.mlp = GemmaMLP(config)
        self.post_cross_layernorm = GemmaRMSNorm(self.hidden_dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        rotary_emb=None,
        output_attentions=False,
    ):
        # 1. Self Attention
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        # Note: We don't usually capture self-attn weights for this specific VLM task, but passing flag is fine
        attn_out, _ = self.self_attn(
            normed_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            output_attentions=False,
        )
        hidden_states = residual + attn_out

        cross_attn_weights = None

        # 2. Cross Attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            normed_states = self.post_attention_layernorm(hidden_states)
            attn_out, cross_attn_weights = self.cross_attn(
                normed_states,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=output_attentions,  # Capture this!
            )
            hidden_states = residual + attn_out

        # 3. MLP
        residual = hidden_states
        normed_states = self.post_cross_layernorm(hidden_states)
        hidden_states = self.mlp(normed_states)
        hidden_states = residual + hidden_states

        return hidden_states, cross_attn_weights
