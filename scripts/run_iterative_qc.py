"""
After training your model once, use it to generate predictions
for Cleanlab Layer 3 analysis
"""

from pathlib import Path

import numpy as np
import torch

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.auto_qc_pipeline import AutoQCPipeline
from data.data_loader import get_dataloader
from model.vlm_model import TrafficVLM


def extract_predictions(model, dataloader, device):
    """Get model predictions for Cleanlab"""
    model.eval()
    all_probs = []

    print("Extracting predictions for Cleanlab...")
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


if __name__ == "__main__":
    # Load trained model
    cfg = ModelConfig()
    t_cfg = TrainingConfig()

    model = TrafficVLM(cfg).to(t_cfg.device)

    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints/vlm_run_01/best_model.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train a model first using: python main.py --experiment_name vlm_run_01")
        exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=t_cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded model from {checkpoint_path}")

    # Get predictions
    train_loader = get_dataloader("train", batch_size=32, shuffle=False)
    predictions = extract_predictions(model, train_loader, t_cfg.device)
    print(f"✓ Extracted {len(predictions)} predictions")

    # Run QC with predictions
    print("\nStarting AutoQC with VLM validation...")
    qc = AutoQCPipeline(use_vlm=True)
    corrected_commands, corrections = qc.run_qc("train", model_predictions=predictions)

    print(f"\n✓ Corrected {len(corrections)} labels")
    print("✓ Saved corrected labels to train_commands_qc.json")
    print("\nNext: Retrain with corrected labels")
    print("  python main.py --experiment_name qc_iteration_2 --epochs 15")
