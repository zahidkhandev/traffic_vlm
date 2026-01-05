# Task 15: Cross-Modal Attention
import math
from typing import Tuple

import torch


class CrossAttentionAnalyzer:
    """
    Task 15: Cross-Attention Analysis & Visualization Tools.
    Helper class to convert raw attention tensors into 2D heatmaps.
    """

    @staticmethod
    def process_attention_maps(
        cross_attention_outputs: Tuple[torch.Tensor],
        image_size: int = 128,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """
        Aggregates cross-attention weights across layers and heads.

        Args:
            cross_attention_outputs: Tuple of tensors from model output.
                                     Each tensor: [Batch, Num_Heads, Text_Seq, Num_Patches]
            image_size: The pixel size of the image (e.g., 128)
            patch_size: The pixel size of a patch (e.g., 16)

        Returns:
            heatmap: [Batch, Text_Seq, Grid_H, Grid_W] - Normalized 0-1 attention map.
        """

        if isinstance(cross_attention_outputs, tuple):
            all_layers = torch.stack(cross_attention_outputs)
        else:
            all_layers = cross_attention_outputs.unsqueeze(0)

        avg_attn = all_layers.mean(dim=0).mean(dim=1)

        batch_size, text_seq, num_patches = avg_attn.shape
        grid_dim = int(math.sqrt(num_patches))

        # [Batch, Text_Seq, Grid, Grid]
        heatmap = avg_attn.view(batch_size, text_seq, grid_dim, grid_dim)

        flat = heatmap.view(batch_size, text_seq, -1)

        min_vals = flat.min(dim=-1, keepdim=True)[0]
        max_vals = flat.max(dim=-1, keepdim=True)[0]
        denom = (max_vals - min_vals) + 1e-9

        flat_norm = (flat - min_vals) / denom

        return flat_norm.view(batch_size, text_seq, grid_dim, grid_dim)
