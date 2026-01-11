import torch
from tqdm import tqdm

from evaluation.metrics import compute_metrics


class Evaluator:
    """
    Handles validation loops and metric computation.
    """

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        """
        Runs inference on the entire validation set.
        Returns: avg_loss, metrics_dict
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        # Loss function for validation tracking
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Running Evaluation"):
                pixel_values = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                outputs = self.model(pixel_values=pixel_values, input_ids=input_ids)
                logits = outputs["logits"]

                # Compute Loss
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Get Predictions
                preds = torch.argmax(logits, dim=-1)

                # Collect
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute Metrics
        metrics = compute_metrics(all_preds, all_labels)
        avg_loss = total_loss / len(self.dataloader)
        metrics["loss"] = avg_loss

        return metrics
