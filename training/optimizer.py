# Task 20: Optimizer Setup

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, learning_rate=1e-4, weight_decay=0.01):
    """
    Optimizer Configuration.
    Separates parameters to apply weight decay correctly (don't decay bias/norms).
    """
    decay_params = []
    no_decay_params = []

    no_decay_names = ["bias", "LayerNorm.weight", "rms_norm.weight"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
