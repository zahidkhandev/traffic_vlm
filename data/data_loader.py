import json
import os

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
        self.images = None

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            # We explicitly retrieve the dataset
            dataset = self.h5_file["images"]
            if isinstance(dataset, h5py.Dataset):
                self.images = dataset
            else:
                raise TypeError("H5 'images' key is not a Dataset.")

        cmd = self.commands[idx]
        image_idx = cmd["image_idx"]

        # Pylance now knows self.images is a Dataset because of the check above
        if self.images is not None:
            image = self.images[image_idx]
        else:
            raise RuntimeError("Images dataset not loaded.")

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor(self.cfg.mean).view(3, 1, 1)
        std = torch.tensor(self.cfg.std).view(3, 1, 1)
        image = (image - mean) / std

        q_text = cmd["q"]
        input_ids = self.tokenizer.encode(q_text, max_len=self.cfg.max_seq_len)

        a_text = cmd["a"]
        answer_ids = self.tokenizer.encode(a_text)
        label_id = answer_ids[1]

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def get_dataloader(split_name, batch_size=4, num_workers=0, shuffle=True):
    """
    Creates a DataLoader for the requested split.
    """
    dataset = TrafficDataset(split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the loader
    loader = get_dataloader("train", batch_size=2)
    print("Testing DataLoader...")

    for batch in loader:
        print(f"Image Batch: {batch['image'].shape}")
        print(f"Text Batch: {batch['input_ids'].shape}")
        print(f"Labels: {batch['label']}")
        break
