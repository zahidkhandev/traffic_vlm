# Task 17: The Master Model Class
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.fusion.cross_attention import CrossAttentionAnalyzer
from model.fusion.multimodal_fusion import MultimodalClassifier
from model.fusion.projection_layer import VisionToLanguageProjection
from model.language.gemma_decoder import GemmaDecoder
from model.vision.siglip_encoder import SigLipVisionTransformer


class TrafficVLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = SigLipVisionTransformer(config)
        self.projector = VisionToLanguageProjection(
            config, vision_hidden_dim=config.hidden_dim
        )
        self.decoder = GemmaDecoder(config)
        self.classifier = MultimodalClassifier(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_vision_features(self, pixel_values):
        vision_outputs = self.vision_encoder(pixel_values)
        projected_vision = self.projector(vision_outputs)
        return projected_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        vision_features = self.get_vision_features(pixel_values)

        decoder_output_dict = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=vision_features,
            attention_mask=attention_mask,
            output_attentions=False,
        )

        decoder_hidden_states = decoder_output_dict["last_hidden_state"]

        logits = self.classifier(decoder_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_cross_attention_map(self, pixel_values, input_ids):
        """
        Extracts visual attention heatmaps.
        Returns: [Batch, Text_Seq, Grid, Grid]
        """
        self.eval()
        with torch.no_grad():
            vision_features = self.get_vision_features(pixel_values)

            decoder_out = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=vision_features,
                output_attentions=True,
            )

            raw_attn = decoder_out.get("cross_attentions")

            if raw_attn is None:
                return None

            heatmap = CrossAttentionAnalyzer.process_attention_maps(
                raw_attn,
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
            )

            return heatmap
