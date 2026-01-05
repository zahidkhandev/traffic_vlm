# Task 9: Patch Embeddings
import torch
import torch.nn as nn

from config.model_config import ModelConfig


class SigLipVisionEmbeddings(nn.Module):
    """
    Constructs the embeddings from the image.

    This module performs the following steps:
    1. Splits the image into patches using a Conv2d layer.
    2. Flattens and transposes the patches.
    3. Adds learnable position embeddings to each patch.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the vision embeddings layer.

        Args:
            config (ModelConfig): Configuration object containing image_size,
                                  patch_size, num_channels, and hidden_dim.
        """
        super().__init__()
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

        # FIX: Define as nn.Embedding so it is callable
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            pixel_values (torch.FloatTensor): Input images of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Patch embeddings of shape [Batch, Num_Patches, Hidden_Dim].
        """
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Now this call works because position_embedding is an nn.Embedding layer
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings
