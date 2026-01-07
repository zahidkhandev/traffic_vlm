import json
import os
from typing import Any

import h5py
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset

from config.dataset_config import DatasetConfig
from data.tokenizer import SimpleTokenizer


class TrafficDataset(Dataset):
    """
    Production PyTorch Dataset for Traffic VLM.
    Integrates heavy augmentation for training to prevent overfitting and memorization.
    """

    def __init__(self, split_name):
        self.cfg = DatasetConfig()
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load_vocab()
        self.split = split_name

        # HEAVY PRODUCTION AUGMENTATION
        # Training: Jitter color/grayscale + Normalize
        if self.split == "train":
            self.transform = T.Compose(
                [
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                    T.RandomGrayscale(p=0.1),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=self.cfg.mean, std=self.cfg.std),
                ]
            )
        # Validation/Test: Just Normalize
        else:
            self.transform = T.Compose(
                [
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

        with open(self.cmd_path, "r") as f:
            self.commands = json.load(f)

        self.h5_file = None
        self.images: Any = None

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        # Open H5 file lazily in the worker process
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]

        # Type check to satisfy strict Pylance
        if self.images is None:
            raise RuntimeError("H5 images dataset failed to load.")

        cmd = self.commands[idx]
        image_idx = cmd["image_idx"]

        # 1. Load image and convert to Tensor [C, H, W]
        # h5py returns numpy array, we convert to torch
        img_raw = torch.from_numpy(self.images[image_idx]).permute(2, 0, 1)

        # 2. Apply augmentation (Jitter/Normalize)
        image = self.transform(img_raw)

        # 3. Tokenize Question
        input_ids = self.tokenizer.encode(cmd["q"], max_len=self.cfg.max_seq_len)

        # 4. Map Answer to Label ID
        a_text = cmd["a"].lower().strip()
        label_id = self.cfg.label_map.get(
            a_text, 1
        )  # Fallback to 1 (No/Unsafe) if unknown

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def get_dataloader(split_name, batch_size=32, num_workers=4, shuffle=True):
    dataset = TrafficDataset(split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Sanity check
    loader = get_dataloader("train", batch_size=2)
    print("Testing DataLoader for 'train' with Augmentation...")
    for batch in loader:
        print(f"Batch Loaded -> Images: {batch['image'].shape}, Labels: {batch['label']}")
        break
