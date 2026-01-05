# Task 32: Masking & Padding Helpers
import numpy as np
import torch


def denormalize_image(tensor):
    """
    Converts a normalized PyTorch tensor (C, H, W) back to a numpy image (H, W, C)
    for visualization.
    Assumes ImageNet normalization means/stds.
    """
    # Clone to avoid modifying the original tensor
    img = tensor.clone().cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverse normalization: img = (tensor * std) + mean
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    # (C, H, W) -> (H, W, C)
    return img.permute(1, 2, 0).numpy()
