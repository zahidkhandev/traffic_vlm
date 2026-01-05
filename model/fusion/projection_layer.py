# Task 14: Vision-to-Language Projection

import torch
import torch.nn as nn

from config.model_config import ModelConfig


class VisionToLanguageProjection(nn.Module):
    """

    Bridges the gap between the Vision Encoder and the Language Decoder.
    If the Vision model outputs 768 dimensions but the Language model
    works in 256 dimensions, this layer performs that translation.
    """

    def __init__(self, config: ModelConfig, vision_hidden_dim: int = 768):
        super().__init__()
        self.vision_dim = vision_hidden_dim
        self.text_dim = config.hidden_dim  # 256

        self.projection = nn.Linear(self.vision_dim, self.text_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

        self.layer_norm = nn.LayerNorm(self.text_dim)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        # 1. Project
        x = self.projection(vision_embeddings)

        # 2. Activate & Dropout
        x = self.activation(x)
        x = self.dropout(x)

        # 3. Normalize
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    # Test
    cfg = ModelConfig()
    projector = VisionToLanguageProjection(cfg, vision_hidden_dim=768)

    dummy_vision = torch.randn(2, 196, 768)  # Batch 2, 196 patches, 768 dim
    output = projector(dummy_vision)

    print(f"Input: {dummy_vision.shape}")
    print(f"Output: {output.shape}")
    # Should be [2, 196, 256]
