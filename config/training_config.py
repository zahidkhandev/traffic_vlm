from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop"""

    batch_size: int = 4
    grad_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 15
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 2
