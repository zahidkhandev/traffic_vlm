import json
import os
from typing import Any  # <--- Added this

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.dataset_config import DatasetConfig
from data.tokenizer import SimpleTokenizer


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for Traffic VLM.
    Loads images from H5 and commands from JSON.
    """

    def __init__(self, split_name):
        self.cfg = DatasetConfig()
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load_vocab()

        cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        if not os.path.exists(cmd_path):
            raise FileNotFoundError(f"Commands file not found: {cmd_path}")

        with open(cmd_path, "r") as f:
            self.commands = json.load(f)

        self.h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Images file not found: {self.h5_path}")

        self.h5_file = None
        self.images: Any = None

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        # Open H5 file once per worker
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]

        # Explicit check prevents runtime errors, 'Any' prevents linter errors
        if self.images is None:
            raise RuntimeError("H5 images dataset failed to load.")

        cmd = self.commands[idx]
        image_idx = cmd["image_idx"]

        # Load Image
        # Pylance will now accept this line because self.images is Any
        image = self.images[image_idx]
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        # Normalize
        mean = torch.tensor(self.cfg.mean).view(3, 1, 1)
        std = torch.tensor(self.cfg.std).view(3, 1, 1)
        image = (image - mean) / std

        # Load Text
        q_text = cmd["q"]
        input_ids = self.tokenizer.encode(q_text, max_len=self.cfg.max_seq_len)

        # Load Label using MAP
        a_text = cmd["a"].lower().strip()

        if a_text in self.cfg.label_map:
            label_id = self.cfg.label_map[a_text]
        else:
            label_id = 1  # Fallback to No/Unsafe

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def get_dataloader(split_name, batch_size=4, num_workers=0, shuffle=True):
    dataset = TrafficDataset(split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    loader = get_dataloader("train", batch_size=2)
    print("Testing DataLoader...")
    for batch in loader:
        print(f"Labels: {batch['label']}")
        break
