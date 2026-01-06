import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config.dataset_config import DatasetConfig
from config.model_config import ModelConfig
from data.data_loader import get_dataloader
from data.tokenizer import SimpleTokenizer
from model.vlm_model import TrafficVLM
from visualization.cross_attention_viz import overlay_attention_heatmap
from visualization.failure_analysis import FailureAnalyzer


def analyze_model(checkpoint_path, output_dir="outputs/analysis"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")

    # 1. Load Model & Data
    config = ModelConfig()
    model = TrafficVLM(config)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load Validation Data (Batch size 1 for easy visualization)
    val_loader = get_dataloader("val", batch_size=1, shuffle=True)
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab()

    print("Generating Visualizations...")

    # --- PART A: Visualize Attention (The "Thinking" Process) ---
    # Get one batch
    batch = next(iter(val_loader))
    pixel_values = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)

    # Get attention map using the method you built in Task 17
    # Note: Requires the 'get_cross_attention_map' method in TrafficVLM
    attn_map = model.get_cross_attention_map(pixel_values, input_ids)

    if attn_map is not None:
        # Take the map for the last token (usually the answer token position)
        # Shape: [Batch, Tokens, H, W] -> [H, W]
        last_token_map = attn_map[0, -1, :, :]

        save_path = os.path.join(output_dir, "attention_sample.png")
        overlay_attention_heatmap(
            pixel_values[0], last_token_map, title="Model Focus", save_path=save_path
        )
        print(f"Attention heatmap saved to {save_path}")
    else:
        print("Model did not return attention maps (check get_cross_attention_map logic)")

    # --- PART B: Failure Analysis (Where did it go wrong?) ---
    print("Running Failure Analysis (this takes a moment)...")
    analyzer = FailureAnalyzer(model, val_loader, device, tokenizer)

    # Find 4 mistakes
    failures = analyzer.find_failures(num_samples=4)

    if failures:
        save_path = os.path.join(output_dir, "failures.png")
        analyzer.visualize_failures(failures, save_path=save_path)
        print(f"Failure grid saved to {save_path}")
    else:
        print("No failures found in the first few batches! Model is doing great.")


if __name__ == "__main__":
    # Point this to your saved checkpoint after training is done
    # Example: checkpoints/vlm_run_01/checkpoint_epoch_0.pt
    # Since you are still training, you can verify with epoch 0 once it saves

    CHECKPOINT = "checkpoints/traffic_vlm_v2/best_model.pt"

    if os.path.exists(CHECKPOINT):
        analyze_model(CHECKPOINT)
    else:
        print("Checkpoint not found yet. Wait for Epoch 1 to finish!")
