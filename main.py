import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config.dataset_config import DatasetConfig

# --- Custom Modules ---
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.data_loader import get_dataloader

# Use your existing classes
from data.tokenizer import SimpleTokenizer
from evaluation.visualization import plot_training_curves
from model.vlm_model import TrafficVLM
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler
from training.trainer import Trainer
from utils.checkpoint_manager import CheckpointManager
from utils.logging_utils import setup_logger


def set_seed(seed):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic VLM Training Pipeline")
    parser.add_argument(
        "--experiment_name", type=str, default="vlm_run_01", help="Name for logging"
    )
    parser.add_argument("--eval_only", action="store_true", help="Run validation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Num epochs")
    return parser.parse_args()


def main():
    # 1. Setup
    args = parse_args()

    # Load configs
    m_config = ModelConfig()
    t_config = TrainingConfig()
    d_config = DatasetConfig()

    # Override config with args
    t_config.batch_size = args.batch_size
    t_config.num_epochs = args.epochs

    # Paths
    output_dir = Path(t_config.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(t_config.checkpoint_dir) / args.experiment_name

    # FIX: Initialize Logger BEFORE using it
    logger = setup_logger("TrafficVLM", save_dir=output_dir)

    # Now it is safe to log
    logger.info(f"Dataset Config: {d_config.__dict__}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Device: {t_config.device}")

    set_seed(42)

    # 2. Data Pipeline
    logger.info("Loading Data...")

    # Tokenizer is loaded internally by TrafficDataset, but we load here to check vocab size if needed
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab()
    logger.info(f"Vocab Size: {len(tokenizer.vocab)}")

    # Use your existing get_dataloader function
    train_loader = get_dataloader(
        "train",
        batch_size=t_config.batch_size,
        num_workers=t_config.num_workers,
        shuffle=True,
    )

    val_loader = get_dataloader(
        "val",
        batch_size=t_config.batch_size,
        num_workers=t_config.num_workers,
        shuffle=False,
    )

    logger.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 3. Model
    logger.info("Initializing Model...")
    model = TrafficVLM(m_config)
    model.to(t_config.device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {param_count / 1e6:.2f}M")

    # 4. Optimizer & Scheduler
    optimizer = get_optimizer(
        model, learning_rate=t_config.learning_rate, weight_decay=t_config.weight_decay
    )

    total_steps = (
        len(train_loader) * t_config.num_epochs // t_config.grad_accumulation_steps
    )
    scheduler = get_scheduler(
        optimizer, num_warmup_steps=t_config.warmup_steps, num_training_steps=total_steps
    )

    # 5. Checkpoint Manager
    ckpt_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    start_epoch = 0
    if args.checkpoint:
        start_epoch, _ = ckpt_manager.load(args.checkpoint, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}")

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=t_config,
        device=t_config.device,
    )

    # 7. Execution
    if args.eval_only:
        logger.info("Evaluation Only...")
        val_loss, val_acc = trainer.validate(epoch_index=0)
        logger.info(f"Final Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        return

    logger.info("Starting Training...")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    try:
        for epoch in range(start_epoch, t_config.num_epochs):
            # Train
            train_loss = trainer.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = trainer.validate(epoch)

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{t_config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )

            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Save
            is_best = val_loss < ckpt_manager.best_metric
            if is_best:
                ckpt_manager.best_metric = val_loss

            ckpt_manager.save(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
                is_best=is_best,
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted.")

    logger.info("Done.")

    # Save curves
    plot_training_curves(history, save_dir=output_dir)


if __name__ == "__main__":
    main()
