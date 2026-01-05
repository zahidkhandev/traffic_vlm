# Task 30: Save/Load Logic
import os
import shutil
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")

    def save(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }

        filename = f"checkpoint_epoch_{epoch}.pt"
        save_path = self.checkpoint_dir / filename
        torch.save(state, save_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copyfile(save_path, best_path)

    def load(self, path, model, optimizer=None, scheduler=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"], checkpoint["metrics"]
