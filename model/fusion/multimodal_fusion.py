# Task 16: Fusion Strategy Logic
import torch
import torch.nn as nn

from config.model_config import ModelConfig


class MultimodalClassifier(nn.Module):
    """
    Task 16: Multimodal Classification Head.

    Takes the final hidden state from the Language Decoder
    and predicts the traffic command response (YES/NO).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_classes = 2  # YES / NO

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
            logits: [Batch, 2]
        """
        # Shape: [Batch, Hidden_Dim]
        last_token_state = hidden_states[:, -1, :]

        # Predict
        logits = self.classifier(last_token_state)

        return logits
