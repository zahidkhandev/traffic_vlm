import torch
import torch.nn.functional as F


def compute_attention_entropy(attention_map):
    """
    Calculates the entropy of the attention distribution.
    Lower entropy = More focused attention.
    """
    # Flatten if 2D
    if len(attention_map.shape) > 2:
        b, h, w = attention_map.shape
        attention_map = attention_map.view(b, -1)  # [Batch, N]

    # Ensure it sums to 1
    attention_map = F.softmax(attention_map, dim=-1)

    # Entropy formula: -sum(p * log(p))
    entropy = -torch.sum(attention_map * torch.log(attention_map + 1e-9), dim=-1)

    return entropy.mean().item()


def compute_pointing_game_accuracy(attention_map, bounding_box_mask):
    """
    Checks if the maximum attention point falls inside the ground truth object box.
    """
    max_idx = torch.argmax(attention_map)
    is_inside = bounding_box_mask.view(-1)[max_idx] == 1
    return is_inside.item()
