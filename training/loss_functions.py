import torch
import torch.nn as nn
import torch.nn.functional as F


class VLMLoss(nn.Module):
    """
    Standard CrossEntropyLoss.

    NOTE: We removed class_weights because we are now using
    WeightedRandomSampler in the DataLoader. Using both would
    cause the model to over-predict minority classes.
    """

    def __init__(self, device: str = "cuda", label_smoothing: float = 0.1):
        super().__init__()

        # Standard Loss (The Sampler handles the balancing)
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.classification_loss(logits, labels)

    @staticmethod
    def auxiliary_contrastive_loss(vision_embeds, text_embeds, temperature=0.07):
        vision_norm = F.normalize(vision_embeds.mean(dim=1), dim=-1)
        text_norm = F.normalize(text_embeds.mean(dim=1), dim=-1)
        logits = torch.matmul(vision_norm, text_norm.t()) / temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)
