# Task 27: Error Analysis
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.dataset_config import DatasetConfig
from utils.tensor_utils import denormalize_image


class FailureAnalyzer:
    def __init__(self, model, dataloader, device, tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.tokenizer = tokenizer
        self.cfg = DatasetConfig()
        self.label_map = {v: k for k, v in self.cfg.label_map.items()}

    def find_failures(self, num_samples=5):
        self.model.eval()
        failures = []

        with torch.no_grad():
            for batch in self.dataloader:
                if len(failures) >= num_samples:
                    break

                pixel_values = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(pixel_values, input_ids)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                incorrect_mask = preds != labels
                incorrect_indices = torch.where(incorrect_mask)[0]

                for idx in incorrect_indices:
                    if len(failures) >= num_samples:
                        break

                    failures.append(
                        {
                            "image": pixel_values[idx].cpu(),
                            "input_ids": input_ids[idx].cpu(),
                            "true_label": labels[idx].item(),
                            "pred_label": preds[idx].item(),
                            "confidence": probs[idx][preds[idx]].item(),
                        }
                    )

        return failures

    def visualize_failures(self, failures, save_path=None):
        n = len(failures)
        if n == 0:
            print("No failures found!")
            return

        cols = min(3, n)  # Reduced columns for better readability of long labels
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(5 * cols, 5 * rows))

        for i, fail in enumerate(failures):
            plt.subplot(rows, cols, i + 1)

            img = denormalize_image(fail["image"])
            img = np.clip(img, 0, 1)

            plt.imshow(img)

            true_name = self.label_map.get(fail["true_label"], "unknown")
            pred_name = self.label_map.get(fail["pred_label"], "unknown")

            title = (
                f"True: {true_name}\nPred: {pred_name}\nConf: {fail['confidence']:.2f}"
            )

            plt.title(title, color="red", fontsize=9)
            plt.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
