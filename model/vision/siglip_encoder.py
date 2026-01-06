# Task 8: SigLip Vision Transformer
import torch
import torch.nn as nn

from config.model_config import ModelConfig


class SigLipVisionEmbeddings(nn.Module):
    """
    Constructs the embeddings from the image.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
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

        # FIX: Missing line added here
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # This will now work because self.position_embedding is defined
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.self_attn_norm = nn.LayerNorm(self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.act = nn.GELU()
        self.mlp_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states):
        residual = hidden_states
        inputs = self.self_attn_norm(hidden_states)

        batch_size, seq_len, _ = inputs.size()

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

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        hidden_states = residual + attn_output

        residual = hidden_states
        inputs = self.mlp_norm(hidden_states)
        inputs = self.fc2(self.act(self.fc1(inputs)))
        hidden_states = residual + inputs

        return hidden_states


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.post_layernorm = nn.LayerNorm(config.hidden_dim)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)
