# Task 18: Main Training Loop

import os

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from training.loss_functions import VLMLoss


class Trainer:
    """
    Task 18: The Main Training Loop.
    Handles the training lifecycle: Forward -> Loss -> Backward -> Optimizer -> Save.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,  # TrainingConfig
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Loss Function
        self.criterion = VLMLoss()

        # Mixed Precision Scaler (Crucial for 6GB VRAM)
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # State tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch_index):
        self.model.train()
        total_loss = 0

        # Progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_index + 1} Training")

        for step, batch in enumerate(progress_bar):
            # 1. Move batch to GPU
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Optional attention mask
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # 2. Mixed Precision Forward Pass
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Model returns logits in dict
                logits = outputs["logits"]

                # Compute Loss
                loss = self.criterion(logits, labels)

                # Normalize loss for gradient accumulation
                # (e.g., if accum=4, we want 1/4th the gradient per step)
                loss = loss / self.config.grad_accumulation_steps

            # 3. Backward Pass (Scaled for AMP)
            self.scaler.scale(loss).backward()

            # 4. Optimizer Step (Delayed for Accumulation)
            if (step + 1) % self.config.grad_accumulation_steps == 0:
                # Clip gradients (prevent explosion)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Step optimizer & scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Step Scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Clear gradients
                self.optimizer.zero_grad()
                self.global_step += 1

            # 5. Logging
            # We multiply back to get the "real" loss for display
            current_loss = loss.item() * self.config.grad_accumulation_steps
            total_loss += current_loss

            # Update TQDM bar
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler
                else self.config.learning_rate
            )
            progress_bar.set_postfix(
                {"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.6f}"}
            )

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch_index):
        """Checks model performance on unseen data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        print(f"\nRunning Validation for Epoch {epoch_index + 1}...")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Standard Forward (No mixed precision needed for eval usually, but safe to keep)
                outputs = self.model(pixel_values=pixel_values, input_ids=input_ids)
                logits = outputs["logits"]

                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Accuracy Calculation
                # Logits shape: [B, 2] -> argmax gives 0 or 1
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        print(
            f"Epoch {epoch_index + 1} Results -> Val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}"
        )
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_loss, val_acc):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        # Standard checkpoint
        filename = f"checkpoints/traffic_vlm_epoch_{epoch + 1}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": self.config,
            },
            filename,
        )

        print(f"Checkpoint saved: {filename}")

        # Save "Best" model based on loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = "checkpoints/traffic_vlm_best.pt"
            torch.save(self.model.state_dict(), best_path)
            print("🏆 New Best Model Saved!")

    def train(self):
        """Runs the full training loop."""
        print(
            f"Starting training for {self.config.num_epochs} epochs on {self.device}..."
        )

        for epoch in range(self.config.num_epochs):
            # 1. Train
            train_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            # 2. Validate
            val_loss, val_acc = self.validate(epoch)

            # 3. Save
            self.save_checkpoint(epoch, val_loss, val_acc)

        print("Training Complete!")
