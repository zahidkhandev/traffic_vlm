# Task 21: LR Scheduling
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Learning Rate Scheduler (Cosine with Warmup).

    Phases:
    1. Linear Warmup: LR goes from 0 to Base_LR
    2. Cosine Decay: LR goes from Base_LR to Min_LR (usually 0)
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
