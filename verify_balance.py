from collections import Counter

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config.dataset_config import DatasetConfig
from data.data_loader import get_dataloader


def verify_dataloader_balance():
    # 1. Get the training loader (which has the WeightedSampler)
    print("Initializing Train Loader...")
    train_loader = get_dataloader(
        split_name="train",
        batch_size=64,
        num_workers=4,
        shuffle=False,  # Shuffle is handled by the sampler
    )

    print(f"Checking {len(train_loader)} batches...")

    # 2. Iterate through one full epoch and count labels
    all_labels = []

    # We don't need gradients or the model, just the data
    for batch in tqdm(train_loader, desc="Scanning Batches"):
        labels = batch["label"].tolist()
        all_labels.extend(labels)

    # 3. Calculate Statistics
    total_samples = len(all_labels)
    counts = Counter(all_labels)

    # Get class names from config for pretty printing
    cfg = DatasetConfig()
    # Invert the map: {0: 'safe', 1: 'no', ...}
    id_to_name = {v: k for k, v in cfg.label_map.items()}

    print(f"\n{'=' * 60}")
    print("EFFECTIVE TRAINING DISTRIBUTION (What the model sees)")
    print(f"{'=' * 60}")
    print(f"{'Class ID':<10} {'Class Name':<20} {'Count':<10} {'Percentage':<10}")
    print(f"{'-' * 60}")

    # Sort by Class ID
    for class_id in sorted(id_to_name.keys()):
        count = counts.get(class_id, 0)
        percentage = (count / total_samples) * 100
        name = id_to_name[class_id]
        print(f"{class_id:<10} {name:<20} {count:<10} {percentage:.2f}%")

    print(f"{'=' * 60}\n")

    # 4. Simple check
    ideal_pct = 100 / len(id_to_name)
    print(f"Ideal Balanced Percentage: ~{ideal_pct:.2f}% per class")
    print("If your values are close to this (e.g., 10-20%), the balancer is working.\n")


if __name__ == "__main__":
    verify_dataloader_balance()
