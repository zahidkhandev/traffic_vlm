# Task 23: Attention Entropy & Grounding
import torch
import torch.nn.functional as F


def compute_attention_entropy(attention_map):
    """
    Calculates the entropy of the attention distribution.
    Lower entropy = More focused attention (Model is looking at something specific).
    Higher entropy = Uniform attention (Model is confused/looking everywhere).

    Args:
        attention_map: [Batch, Patches] or [Batch, H, W] (normalized 0-1)
    """
    # Flatten if 2D
    if len(attention_map.shape) > 2:
        b, h, w = attention_map.shape
        attention_map = attention_map.view(b, -1)  # [Batch, N]

    # Ensure it sums to 1 (probability distribution)
    attention_map = F.softmax(attention_map, dim=-1)

    # Entropy formula: -sum(p * log(p))
    # Add epsilon to avoid log(0)
    entropy = -torch.sum(attention_map * torch.log(attention_map + 1e-9), dim=-1)

    return entropy.mean().item()


def compute_pointing_game_accuracy(attention_map, bounding_box_mask):
    """
    checks if the maximum attention point falls inside the ground truth object box.
    (Requires bounding box masks for validation data).

    Args:
        attention_map: [H, W] heatmap
        bounding_box_mask: [H, W] binary mask (1 inside box, 0 outside)
    """
    # Find coordinate of max attention
    # This is a simple implementation of the "Pointing Game" metric
    max_idx = torch.argmax(attention_map)

    # Check if that index is 1 in the mask
    # Flatten mask to match argmax index
    is_inside = bounding_box_mask.view(-1)[max_idx] == 1

    return is_inside.item()
