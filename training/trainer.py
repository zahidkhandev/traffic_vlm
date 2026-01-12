import logging
import os

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from training.loss_functions import VLMLoss

logger = logging.getLogger("TrafficVLM")


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.criterion = VLMLoss(device=self.device)
        self.scaler = GradScaler("cuda", enabled=config.mixed_precision)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def train_one_epoch(self, epoch_index):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_index + 1} Training")

        for step, batch in enumerate(progress_bar):
            pixel_values = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast("cuda", enabled=self.config.mixed_precision):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                )
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)
                loss = loss / self.config.grad_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

                self.optimizer.zero_grad()

                if self.scheduler:
                    if scale_after >= scale_before:
                        self.scheduler.step()
                        self.global_step += 1

            current_loss = loss.item() * self.config.grad_accumulation_steps
            total_loss += current_loss

            lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler
                else self.config.learning_rate
            )
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{lr:.6f}"})

        return total_loss / len(self.train_loader)

    def validate(self, epoch_index):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        logger.info(f"\nRunning Validation for Epoch {epoch_index + 1}...")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(pixel_values=pixel_values, input_ids=input_ids)
                logits = outputs["logits"]

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        # --- UPDATED: Uses logger instead of print ---
        classes = ["Safe", "Red Light", "Pedestrian", "Vehicle", "Obstacle"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EPOCH {epoch_index + 1} BREAKDOWN")
        logger.info(f"{'=' * 60}")
        logger.info(f"{'Class':<20} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
        logger.info(f"{'-' * 60}")

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        for i, class_name in enumerate(classes):
            indices = all_labels == i
            class_total = indices.sum()

            if class_total > 0:
                class_correct = (all_preds[indices] == i).sum()
                class_acc = class_correct / class_total
                logger.info(
                    f"{class_name:<20} {class_total:<10} {class_correct:<10} {class_acc:.2%}"
                )
            else:
                logger.info(f"{class_name:<20} {0:<10} {0:<10} N/A")
        logger.info(f"{'=' * 60}\n")

        logger.info(f"Overall Val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_loss, val_acc):
        checkpoint_dir = os.path.join(
            "checkpoints", getattr(self.config, "run_name", "default_run")
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        filename = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(state, filename)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            torch.save(state, os.path.join(checkpoint_dir, "best_model.pt"))
            logger.info(f"[SAVED BEST] {val_acc:.2%}")

    def load_checkpoint(self, checkpoint_path):
        pass

    def train(self, resume_from=None):
        start_epoch = 0

        logger.info(f"Starting training on {self.device}...")

        for epoch in range(start_epoch, self.config.num_epochs):
            train_loss = self.train_one_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            val_loss, val_acc = self.validate(epoch)
            self.save_checkpoint(epoch, val_loss, val_acc)
