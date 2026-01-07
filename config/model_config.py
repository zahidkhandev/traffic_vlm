from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for the Traffic VLM (Nano Architecture).
    """

    image_size: int = 128  # 128x128 pixels
    patch_size: int = 16  # 16x16 pixel patches
    num_channels: int = 3  # RGB images
    hidden_dim: int = 256  # Embedding dimension

    num_layers: int = 4  # Depth of the Transformer
    num_heads: int = 8  # 256 dim / 8 heads = 32 dim per head (Standard ratio)

    vocab_size: int = 500  # Small vocab for specific traffic commands
    max_seq_len: int = 32  # Max length of questions/answers
    intermediate_size: int = 1024  # 4x hidden_dim (256 * 4)

    dropout: float = 0.1  # 10% dropout to prevent overfitting

    num_classes: int = 6
