# Task 26: Cross-Attention Heatmaps
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.tensor_utils import denormalize_image


def overlay_attention_heatmap(
    image_tensor, attention_map, alpha=0.6, title="Cross Attention", save_path=None
):
    """
    Overlays a 2D attention map onto an image.
    """
    # 1. Prepare Image
    img_np = denormalize_image(image_tensor)  # [H, W, 3]
    h, w = img_np.shape[:2]

    # 2. Prepare Heatmap
    if isinstance(attention_map, torch.Tensor):
        attn = attention_map.detach().cpu().numpy()
    else:
        attn = attention_map

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(attn, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize to 0-255 for color mapping
    # FIX: Use .astype(np.uint8) to satisfy Pylance type checker regarding ndarray
    heatmap_norm = (255 * heatmap_resized).astype(np.uint8)

    # Apply colormap (JET is standard for attention)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # 3. Overlay
    overlay = (1 - alpha) * img_np + alpha * heatmap_color

    # 4. Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
