from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.language.decoder_layer import GemmaDecoderLayer, GemmaRMSNorm
from model.language.rope_embeddings import RotaryEmbedding


class GemmaDecoder(nn.Module):
    """
    The Main Language Decoder.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size

        # 1. Text Embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=self.padding_idx
        )

        # 2. RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_dim // config.num_heads,
            max_position_embeddings=config.max_seq_len * 2,
        )

        # 3. Layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_layers)]
        )

        # 4. Norm
        self.norm = GemmaRMSNorm(config.hidden_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        hidden_states = self.embed_tokens(input_ids)

        all_cross_attentions = ()

        for layer in self.layers:
            hidden_states, layer_cross_attn = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                rotary_emb=self.rotary_emb,
                output_attentions=output_attentions,
            )

            if output_attentions and layer_cross_attn is not None:
                all_cross_attentions = all_cross_attentions + (layer_cross_attn,)

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "cross_attentions": all_cross_attentions if output_attentions else None,
        }
