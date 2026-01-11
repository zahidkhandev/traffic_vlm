import json
import os
from typing import Any, List

import h5py
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config.dataset_config import DatasetConfig
from data.tokenizer import SimpleTokenizer


class TrafficDataset(Dataset):
    """
    Production PyTorch Dataset for Traffic VLM.
    Integrates heavy augmentation and SAFE filtering.
    """

    def __init__(self, split_name):
        self.cfg = DatasetConfig()
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load_vocab()
        self.split = split_name

        # --- HEAVY PRODUCTION AUGMENTATION ---
        # This creates "new" data every epoch by modifying existing images.
        if self.split == "train":
            self.transform = T.Compose(
                [
                    # 1. Geometry: Zoom & Rotate (No Skew)
                    # Simulates different distances and camera vibrations
                    T.RandomResizedCrop(
                        size=(self.cfg.image_size, self.cfg.image_size),
                        scale=(0.85, 1.0),  # Zoom in slightly (15% max)
                        ratio=(0.9, 1.1),  # Keep aspect ratio mostly square
                        antialias=True,
                    ),
                    T.RandomRotation(degrees=(-5, 5)),
                    # 2. Photometric: Heavy Color/Lighting (The "Colors" part)
                    T.ColorJitter(
                        brightness=0.5,  # Drastic lighting changes (Shadows/Sun)
                        contrast=0.5,  # Washed out vs High Contrast
                        saturation=0.5,  # Black&White vs Oversaturated
                        hue=0.05,  # Slight hue shift (Red stays Red)
                    ),
                    T.RandomGrayscale(p=0.1),  # Occasional B&W cam simulation
                    # 3. Quality & Sensor Noise
                    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Sharp/Blurry
                    T.RandomAutocontrast(p=0.3),
                    T.RandomApply(
                        [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.2
                    ),
                    # 4. Standard Tensor Conversion
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=self.cfg.mean, std=self.cfg.std),
                    # 5. Regularization: Cutout (Crucial for small datasets)
                    # Forces model to look at "road" if "car" is blocked by a black box
                    T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
                ]
            )
        # Validation/Test: Deterministic (No Randomness)
        else:
            self.transform = T.Compose(
                [
                    T.Resize((self.cfg.image_size, self.cfg.image_size), antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=self.cfg.mean, std=self.cfg.std),
                ]
            )

        # Load file paths
        self.cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        self.h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")

        if not os.path.exists(self.cmd_path):
            raise FileNotFoundError(f"Commands file not found: {self.cmd_path}")

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        # --- FILTERING & LABEL PRE-CALCULATION ---
        with open(self.cmd_path, "r") as f:
            all_cmds = json.load(f)

        self.commands = [c for c in all_cmds if c.get("type") == "safety"]

        # Pre-calculate labels for WeightedRandomSampler
        self.labels_indices: List[int] = []
        for cmd in self.commands:
            a_text = cmd["a"].lower().strip()
            label_id = self.cfg.label_map.get(a_text, 1)  # Default to 'no' if unknown
            self.labels_indices.append(label_id)

        print(
            f"Loaded {len(self.commands)} Safety commands from {self.split} (Filtered out {len(all_cmds) - len(self.commands)} aux items)"
        )

        self.h5_file = None
        self.images: Any = None

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        # Open H5 file lazily in the worker process
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]

        if self.images is None:
            raise RuntimeError("H5 images dataset failed to load.")

        cmd = self.commands[idx]
        image_idx = cmd["image_idx"]

        # 1. Load image and convert to Tensor [C, H, W]
        img_raw = torch.from_numpy(self.images[image_idx]).permute(2, 0, 1)

        # 2. Apply augmentation
        image = self.transform(img_raw)

        # 3. Tokenize Question
        input_ids = self.tokenizer.encode(cmd["q"], max_len=self.cfg.max_seq_len)

        # 4. Get Pre-calculated Label
        label_id = self.labels_indices[idx]

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def get_dataloader(split_name, batch_size=32, num_workers=4, shuffle=True):
    dataset = TrafficDataset(split_name)

    sampler = None

    # --- BALANCED DATA LOADING LOGIC ---
    # This is what "Increases" the minority classes.
    # It will pick "Stop" images way more often than "Safe" images.
    if split_name == "train":
        print("[INFO] Computing class weights for Balanced Sampling...")

        # Convert list to tensor for faster operations
        targets = torch.tensor(dataset.labels_indices)

        # Count samples per class
        class_counts = torch.bincount(targets)

        # Calculate Weight: (1 / Count). Rare classes get HUGE weights.
        class_weights = 1.0 / (class_counts.float() + 1e-6)

        # Assign weight to each individual sample based on its class
        sample_weights = class_weights[targets]

        print(f"[INFO] Class Counts: {class_counts.tolist()}")
        print(f"[INFO] Class Weights: {class_weights.tolist()}")

        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),  # Convert to Python list
            num_samples=len(sample_weights),  # Total size stays same (Epoch size)
            replacement=True,  # Allow picking same image multiple times (Augmentation makes it unique)
        )

        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Sanity check
    loader = get_dataloader("train", batch_size=10)
    print("Testing DataLoader for 'train' with WeightedSampler & Augmentation...")

    counts = {}
    for i, batch in enumerate(loader):
        labels = batch["label"].tolist()
        print(f"Batch {i} Labels: {labels}")
        print(f"Batch {i} Image Shape: {batch['image'].shape}")

        for label_val in labels:
            counts[label_val] = counts.get(label_val, 0) + 1

        if i >= 2:
            break

    print(f" Sample Distribution: {counts}")
