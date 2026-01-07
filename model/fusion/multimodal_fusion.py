# Task 16: Fusion Strategy Logic
import torch
import torch.nn as nn

from config.model_config import ModelConfig


class MultimodalClassifier(nn.Module):
    """
    Multimodal Classification Head.

    Takes the final hidden state from the Language Decoder
    and predicts the traffic command response (Safe, Red Light, Pedestrian, etc.).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_classes = config.num_classes

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Dim]
        Returns:
            logits: [Batch, Num_Classes]
        """
        # Shape: [Batch, Hidden_Dim]
        last_token_state = hidden_states[:, -1, :]

        # Predict
        logits = self.classifier(last_token_state)

        return logits
