from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop"""

    # Optimization
    batch_size: int = 4
    grad_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    num_epochs: int = 15

    # Scheduler
    warmup_steps: int = 500

    # System
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 2

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
