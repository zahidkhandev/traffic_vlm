# Task 19: Loss Definitions

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLMLoss(nn.Module):
    """
    Loss Functions.
    Defines how we penalize the model when it's wrong.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [Batch, 2] (Raw scores for NO/YES)
            labels: [Batch] (0 or 1)
        """
        return self.classification_loss(logits, labels)

    @staticmethod
    def auxiliary_contrastive_loss(vision_embeds, text_embeds, temperature=0.07):
        """
        Forces image features and text features to align.
        Useful if the model struggles to converge.
        """
        # Normalize
        vision_norm = F.normalize(vision_embeds.mean(dim=1), dim=-1)
        text_norm = F.normalize(text_embeds.mean(dim=1), dim=-1)

        # Cosine similarity
        logits = torch.matmul(vision_norm, text_norm.t()) / temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        return F.cross_entropy(logits, labels)
