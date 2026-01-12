import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config.dataset_config import DatasetConfig
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.data_loader import get_dataloader
from data.tokenizer import SimpleTokenizer
from evaluation.visualization import plot_training_curves
from model.vlm_model import TrafficVLM
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler
from training.trainer import Trainer
from utils.checkpoint_manager import CheckpointManager
from utils.logging_utils import setup_logger
from visualization.cross_attention_viz import overlay_attention_heatmap
from visualization.failure_analysis import FailureAnalyzer


def set_seed(seed):
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
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to best model weights for fine-tuning",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Num epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override Learning Rate")
    return parser.parse_args()


def visualize_epoch(model, val_loader, tokenizer, device, epoch, output_dir):
    viz_dir = output_dir / "visualizations" / f"epoch_{epoch + 1}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    try:
        batch = next(iter(val_loader))
        pixel_values = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)

        num_samples = min(4, pixel_values.size(0))

        if hasattr(model, "get_cross_attention_map"):
            attn_maps = model.get_cross_attention_map(pixel_values, input_ids)

            if attn_maps is not None:
                for i in range(num_samples):
                    heatmap = attn_maps[i, -1, :, :]
                    save_path = viz_dir / f"attention_sample_{i}.png"
                    overlay_attention_heatmap(
                        pixel_values[i],
                        heatmap,
                        title=f"Epoch {epoch + 1} Sample {i}",
                        save_path=str(save_path),
                    )
    except Exception as e:
        print(f"Visualization Warning: {e}")

    try:
        analyzer = FailureAnalyzer(model, val_loader, device, tokenizer)
        failures = analyzer.find_failures(num_samples=9)

        if failures:
            save_path = viz_dir / "failures.png"
            analyzer.visualize_failures(failures, save_path=str(save_path))
    except Exception as e:
        print(f"Visualization Warning: {e}")


def main():
    args = parse_args()

    m_config = ModelConfig()
    t_config = TrainingConfig()
    d_config = DatasetConfig()

    t_config.batch_size = args.batch_size
    t_config.num_epochs = args.epochs

    if args.lr:
        t_config.learning_rate = args.lr

    output_dir = Path(t_config.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(t_config.checkpoint_dir) / args.experiment_name

    logger = setup_logger("TrafficVLM", save_dir=output_dir)

    logger.info(f"Dataset Config: {d_config.__dict__}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Device: {t_config.device}")
    logger.info(f"Batch Size: {t_config.batch_size}")
    logger.info(f"Learning Rate: {t_config.learning_rate}")

    set_seed(42)

    logger.info("Loading Data...")

    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab()
    logger.info(f"Vocab Size: {len(tokenizer.vocab)}")

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

    logger.info("Initializing Model...")
    model = TrafficVLM(m_config)
    model.to(t_config.device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {param_count / 1e6:.2f}M")

    optimizer = get_optimizer(
        model, learning_rate=t_config.learning_rate, weight_decay=t_config.weight_decay
    )

    total_steps = (
        len(train_loader) * t_config.num_epochs // t_config.grad_accumulation_steps
    )
    scheduler = get_scheduler(
        optimizer, num_warmup_steps=t_config.warmup_steps, num_training_steps=total_steps
    )

    ckpt_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    start_epoch = 0

    if args.checkpoint:
        start_epoch, _ = ckpt_manager.load(args.checkpoint, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}")

    elif args.weights:
        logger.info(f"Loading weights for FINE-TUNING from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=t_config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = 0
        logger.info(
            "Weights loaded. Optimizer and Scheduler reset. Starting fresh from Epoch 1."
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=t_config,
        device=t_config.device,
    )

    if args.eval_only:
        logger.info("Evaluation Only...")
        val_loss, val_acc = trainer.validate(epoch_index=0)
        logger.info(f"Final Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        return

    logger.info("Starting Training...")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    try:
        for epoch in range(start_epoch, t_config.num_epochs):
            train_loss = trainer.train_one_epoch(epoch)
            val_loss, val_acc = trainer.validate(epoch)

            logger.info(
                f"Epoch {epoch + 1}/{t_config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

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

            plot_training_curves(history, save_dir=output_dir)
            visualize_epoch(
                model, val_loader, tokenizer, t_config.device, epoch, output_dir
            )
            logger.info(
                f"Visualizations saved to {output_dir}/visualizations/epoch_{epoch + 1}"
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted.")

    logger.info("Done.")
    plot_training_curves(history, save_dir=output_dir)


if __name__ == "__main__":
    main()
