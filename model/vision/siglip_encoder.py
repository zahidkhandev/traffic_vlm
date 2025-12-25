# Task 8: SigLip Vision Transformer
import torch
from torch import nn

from config.model_config import ModelConfig


class SigLipVisionEmbeddings(nn.Module):
    """
    Takes an image and chops it into patches.
    """

    def __init__(self, config: ModelConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.embed_dim = config.hidden_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SigLipEncoderLayer(nn.Module):
    """
    Looking at patches and relating them to each other.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # 1 / sqrt(dim)

        # --- Self Attention (The "Looking" part) ---
        self.self_attn_norm = nn.LayerNorm(self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # --- Feed Forward (The "Processing" part) ---
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4)  # Expand 4x
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)  # Compress back
        self.act = nn.GELU()  # Activation function
        self.mlp_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states):
        # --- 1. Self Attention Block ---
        residual = hidden_states
        # Norm BEFORE attention (Pre-Norm architecture is more stable)
        inputs = self.self_attn_norm(hidden_states)

        batch_size, seq_len, _ = inputs.size()

        # Project Q, K, V
        query = (
            self.q_proj(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Calculate Scores: Q * K^T
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Apply Attention: Scores * V
        attn_output = torch.matmul(attn_weights, value)

        # Reassemble Heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        # Add Residual (Skip Connection)
        hidden_states = residual + attn_output

        # --- 2. Feed Forward Block ---
        residual = hidden_states
        inputs = self.mlp_norm(hidden_states)

        inputs = self.fc1(inputs)
        inputs = self.act(inputs)
        inputs = self.fc2(inputs)

        # Add Residual
        hidden_states = residual + inputs

        return hidden_states


class SigLipVisionTransformer(nn.Module):
    """
    Stacks Embeddings + N Encoder Layers.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. The Retina
        self.embeddings = SigLipVisionEmbeddings(config)

        # 2. The Brain Layers (Stacked)
        self.encoder = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_layers)]
        )

        # 3. Final cleanup norm
        self.post_layernorm = nn.LayerNorm(config.hidden_dim)

    def forward(self, pixel_values):
        # Convert pixels to tokens [B, 64, 256]
        hidden_states = self.embeddings(pixel_values)

        # Pass through each Transformer layer
        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        # Final Norm
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


if __name__ == "__main__":
    # Test the Encoder
    print("Testing Vision Encoder...")
    cfg = ModelConfig()
    model = SigLipVisionTransformer(cfg)

    # Fake image batch: 2 images, 3 channels, 128x128
    dummy_img = torch.randn(2, 3, 128, 128)
    output = model(dummy_img)

    print(f"Input Shape: {dummy_img.shape}")
    print(f"Output Shape: {output.shape}")
    print("Expected Output: [2, 64, 256] (Batch, Num_Patches, Hidden_Dim)")
