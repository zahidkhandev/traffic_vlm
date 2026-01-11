import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.vlm_model import TrafficVLM
from visualization.cross_attention_viz import overlay_attention_heatmap


class ComprehensiveModelAnalyzer:
    """Comprehensive analysis tool for Traffic VLM models."""

    def __init__(
        self,
        checkpoint_path: str,
        run_name: str,
        test_split: str = "test",
        device: Optional[str] = None,
        custom_dataloader_fn=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.run_name = run_name
        self.test_split = test_split
        self.device: str = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.custom_dataloader_fn = custom_dataloader_fn

        self.output_dir = Path("outputs") / run_name / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 80)
        print(f"Run Name: {run_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test Split: {test_split}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)

        self._load_configs()
        self._load_model()
        self._load_data()

        self.all_predictions: List[int] = []
        self.all_labels: List[int] = []
        self.all_probs: List[np.ndarray] = []
        self.failure_cases: List[Dict] = []
        self.success_cases: List[Dict] = []

    def _load_configs(self):
        """Load configurations from the saved run directory."""
        print("\n[1/6] Loading Configurations from run directory...")

        run_config_dir = Path("outputs") / self.run_name

        model_config_path = run_config_dir / "model_config.py"
        if model_config_path.exists():
            spec = spec_from_file_location("model_config", model_config_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {model_config_path}")
            model_config_module = module_from_spec(spec)
            spec.loader.exec_module(model_config_module)
            self.model_config = model_config_module.ModelConfig()
            print(f"  ✓ Loaded ModelConfig from {model_config_path}")
        else:
            raise FileNotFoundError(f"ModelConfig not found at {model_config_path}")

        dataset_config_path = run_config_dir / "dataset_config.py"
        if dataset_config_path.exists():
            spec = spec_from_file_location("dataset_config", dataset_config_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {dataset_config_path}")
            dataset_config_module = module_from_spec(spec)
            spec.loader.exec_module(dataset_config_module)
            self.dataset_config = dataset_config_module.DatasetConfig()
            print(f"  ✓ Loaded DatasetConfig from {dataset_config_path}")
        else:
            raise FileNotFoundError(f"DatasetConfig not found at {dataset_config_path}")

        self.num_classes = self.model_config.num_classes
        print(f"  ✓ Number of Classes: {self.num_classes}")
        print(f"  ✓ Image Size: {self.model_config.image_size}")
        print(f"  ✓ Patch Size: {self.model_config.patch_size}")
        print(f"  ✓ Vocab Size: {self.model_config.vocab_size}")

        self.id_to_label = {v: k for k, v in self.dataset_config.label_map.items()}
        print(f"  ✓ Label Map: {self.id_to_label}")

    def _load_model(self):
        print("\n[2/6] Loading Model...")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model = TrafficVLM(self.model_config)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "epoch" in checkpoint:
                print(f"  ✓ Loaded checkpoint from Epoch {checkpoint['epoch']}")
            if "val_acc" in checkpoint:
                print(f"  ✓ Checkpoint Val Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print("  ✓ Model loaded and set to evaluation mode")

    def _load_data(self):
        print("\n[3/6] Loading Data...")

        if self.custom_dataloader_fn is None:
            raise RuntimeError("Custom dataloader function is required!")

        self.test_loader = self.custom_dataloader_fn(
            self.test_split, batch_size=1, shuffle=False, num_workers=0
        )
        print(f"  ✓ Loaded {len(self.test_loader)} {self.test_split} samples")

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        h5_path = Path(self.dataset_config.output_dir) / f"{self.test_split}.h5"
        if not h5_path.exists():
            print(f"  ⚠ Metadata file not found: {h5_path}")
            return []

        metadata = []
        with h5py.File(h5_path, "r") as hf:
            meta_ds = hf["metadata"]
            if isinstance(meta_ds, h5py.Dataset):
                for idx in range(len(meta_ds)):
                    raw_data = meta_ds[idx]
                    if isinstance(raw_data, bytes):
                        json_str = raw_data.decode("utf-8")
                    else:
                        json_str = str(raw_data)
                    metadata.append(json.loads(json_str))

        print(f"  ✓ Loaded metadata for {len(metadata)} samples")
        return metadata

    def run_inference(self, max_samples=None):
        print("\n[4/6] Running Inference...")

        self.model.eval()

        total_samples = max_samples if max_samples else len(self.test_loader)

        with torch.no_grad():
            for idx, batch in enumerate(
                tqdm(self.test_loader, desc="  Processing", total=total_samples)
            ):
                if max_samples and len(self.all_predictions) >= max_samples:
                    break

                if max_samples and idx > max_samples * 10:
                    print(f"\n  ⚠ Reached max iteration limit at {idx} samples")
                    break

                pixel_values = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"]

                outputs = self.model(pixel_values, input_ids)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)

                pred_id = int(pred.item())
                label_id = int(labels.item())
                prob_dist = probs[0].cpu().numpy()

                self.all_predictions.append(pred_id)
                self.all_labels.append(label_id)
                self.all_probs.append(prob_dist)

                case_data = {
                    "idx": idx,
                    "image": pixel_values[0].cpu(),
                    "pred": pred_id,
                    "label": label_id,
                    "probs": prob_dist,
                    "confidence": float(prob_dist[pred_id]),
                }

                if pred_id != label_id:
                    self.failure_cases.append(case_data)
                else:
                    self.success_cases.append(case_data)

        print("  ✓ Inference complete")
        print(f"  ✓ Total Samples: {len(self.all_predictions)}")
        print(f"  ✓ Failures: {len(self.failure_cases)}")
        print(f"  ✓ Successes: {len(self.success_cases)}")

    def compute_metrics(self) -> Dict:
        print("\n[5/6] Computing Metrics...")

        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

        accuracy = float(accuracy_score(y_true, y_pred) * 100)
        precision_macro = float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0, labels=unique_labels
            )
            * 100
        )
        recall_macro = float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0, labels=unique_labels
            )
            * 100
        )
        f1_macro = float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0, labels=unique_labels
            )
            * 100
        )

        precision_weighted = float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=unique_labels
            )
            * 100
        )
        recall_weighted = float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=unique_labels
            )
            * 100
        )
        f1_weighted = float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=unique_labels
            )
            * 100
        )

        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0, labels=unique_labels
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        tp_per_class = np.diag(cm)
        fp_per_class = cm.sum(axis=0) - tp_per_class
        fn_per_class = cm.sum(axis=1) - tp_per_class
        tn_per_class = cm.sum() - (tp_per_class + fp_per_class + fn_per_class)

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "unique_labels": unique_labels,
            "per_class": {
                "precision": precision_per_class * 100,
                "recall": recall_per_class * 100,
                "f1": f1_per_class * 100,
                "support": support_per_class,
                "tp": tp_per_class,
                "fp": fp_per_class,
                "fn": fn_per_class,
                "tn": tn_per_class,
            },
        }

        print(f"  ✓ Overall Accuracy: {accuracy:.2f}%")
        print(f"  ✓ Macro F1-Score: {f1_macro:.2f}%")
        print(f"  ✓ Weighted F1-Score: {f1_weighted:.2f}%")
        print(f"  ✓ Classes present in test set: {unique_labels}")

        return metrics

    def generate_visualizations(self, metrics: Dict):
        print("\n[6/6] Generating Visualizations...")

        self._plot_confusion_matrix(metrics["confusion_matrix"], metrics["unique_labels"])
        print("  ✓ Saved confusion_matrix.png")

        self._plot_per_class_metrics(metrics["per_class"], metrics["unique_labels"])
        print("  ✓ Saved per_class_metrics.png")

        self._plot_confusion_matrix_normalized(
            metrics["confusion_matrix"], metrics["unique_labels"]
        )
        print("  ✓ Saved confusion_matrix_normalized.png")

        self._plot_tp_tn_fp_fn(metrics["per_class"], metrics["unique_labels"])
        print("  ✓ Saved tp_tn_fp_fn_breakdown.png")

        self._plot_confidence_distribution()
        print("  ✓ Saved confidence_distribution.png")

        self._plot_failure_cases()
        print("  ✓ Saved failure_cases_grid.png")

        self._plot_high_confidence_failures()
        print("  ✓ Saved high_confidence_failures.png")

        self._plot_class_distribution(metrics["unique_labels"])
        print("  ✓ Saved class_distribution.png")

        self._plot_attention_failures()
        print("  ✓ Saved attention_failure_*.png")

        self._plot_confidence_vs_accuracy()
        print("  ✓ Saved confidence_vs_accuracy.png")

    def _plot_confusion_matrix(self, cm: np.ndarray, unique_labels: list):
        fig, ax = plt.subplots(figsize=(10, 8))

        class_labels = [self.id_to_label.get(i, f"Class_{i}") for i in unique_labels]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Confusion Matrix - {self.run_name}", fontsize=14, fontweight="bold"
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_confusion_matrix_normalized(self, cm: np.ndarray, unique_labels: list):
        fig, ax = plt.subplots(figsize=(10, 8))

        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        class_labels = [self.id_to_label.get(i, f"Class_{i}") for i in unique_labels]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="YlOrRd",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={"label": "Percentage"},
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Normalized Confusion Matrix - {self.run_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix_normalized.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_per_class_metrics(self, per_class: Dict, unique_labels: list):
        fig, ax = plt.subplots(figsize=(12, 6))

        class_labels = [self.id_to_label.get(i, f"Class_{i}") for i in unique_labels]
        x = np.arange(len(class_labels))
        width = 0.25

        ax.bar(x - width, per_class["precision"], width, label="Precision", alpha=0.8)
        ax.bar(x, per_class["recall"], width, label="Recall", alpha=0.8)
        ax.bar(x + width, per_class["f1"], width, label="F1-Score", alpha=0.8)

        ax.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Per-Class Metrics - {self.run_name}", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "per_class_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_tp_tn_fp_fn(self, per_class: Dict, unique_labels: list):
        fig, ax = plt.subplots(figsize=(14, 6))

        class_labels = [self.id_to_label.get(i, f"Class_{i}") for i in unique_labels]
        x = np.arange(len(class_labels))
        width = 0.2

        ax.bar(
            x - 1.5 * width,
            per_class["tp"],
            width,
            label="True Positives",
            color="green",
            alpha=0.8,
        )
        ax.bar(
            x - 0.5 * width,
            per_class["tn"],
            width,
            label="True Negatives",
            color="blue",
            alpha=0.8,
        )
        ax.bar(
            x + 0.5 * width,
            per_class["fp"],
            width,
            label="False Positives",
            color="orange",
            alpha=0.8,
        )
        ax.bar(
            x + 1.5 * width,
            per_class["fn"],
            width,
            label="False Negatives",
            color="red",
            alpha=0.8,
        )

        ax.set_xlabel("Class (One-vs-Rest)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.set_title(
            f"TP/TN/FP/FN Breakdown - {self.run_name}", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "tp_tn_fp_fn_breakdown.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_confidence_distribution(self):
        correct_conf = [case["confidence"] for case in self.success_cases]
        incorrect_conf = [case["confidence"] for case in self.failure_cases]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            correct_conf,
            bins=50,
            alpha=0.6,
            label="Correct",
            color="green",
            edgecolor="black",
        )
        ax.hist(
            incorrect_conf,
            bins=50,
            alpha=0.6,
            label="Incorrect",
            color="red",
            edgecolor="black",
        )

        ax.set_xlabel("Confidence (Probability)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Confidence Distribution - {self.run_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_failure_cases(self, max_samples: int = 20):
        if not self.failure_cases:
            print("  ⚠ No failure cases to visualize")
            return

        sorted_failures = sorted(
            self.failure_cases, key=lambda x: x["confidence"], reverse=True
        )
        samples_to_plot = sorted_failures[:max_samples]

        n_samples = len(samples_to_plot)
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, case in enumerate(samples_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = case["image"].permute(1, 2, 0).numpy()
            img = img * np.array(self.dataset_config.std) + np.array(
                self.dataset_config.mean
            )
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            pred_label = self.id_to_label.get(case["pred"], f"Class_{case['pred']}")
            true_label = self.id_to_label.get(case["label"], f"Class_{case['label']}")

            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {case['confidence']:.2%}",
                fontsize=8,
                color="red",
            )
            ax.axis("off")

        for idx in range(n_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle(
            f"Failure Cases (Top {n_samples} by Confidence)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "failure_cases_grid.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_high_confidence_failures(
        self, threshold: float = 0.8, max_samples: int = 12
    ):
        high_conf_failures = [
            c for c in self.failure_cases if c["confidence"] >= threshold
        ]

        if not high_conf_failures:
            print(f"  ⚠ No high-confidence failures (threshold={threshold})")
            return

        samples_to_plot = high_conf_failures[:max_samples]
        n_samples = len(samples_to_plot)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, case in enumerate(samples_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = case["image"].permute(1, 2, 0).numpy()
            img = img * np.array(self.dataset_config.std) + np.array(
                self.dataset_config.mean
            )
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            pred_label = self.id_to_label.get(case["pred"], f"Class_{case['pred']}")
            true_label = self.id_to_label.get(case["label"], f"Class_{case['label']}")

            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {case['confidence']:.2%}",
                fontsize=8,
                color="darkred",
                fontweight="bold",
            )
            ax.axis("off")

        for idx in range(n_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle(
            f"High-Confidence Failures (>={threshold:.0%})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "high_confidence_failures.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_class_distribution(self, unique_labels: list):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        unique_true, counts_true = np.unique(self.all_labels, return_counts=True)
        ax1.bar(
            range(len(unique_true)),
            counts_true,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("True Label Distribution", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(len(unique_true)))
        ax1.set_xticklabels(
            [self.id_to_label.get(i, f"Class_{i}") for i in unique_true],
            rotation=45,
            ha="right",
        )
        ax1.grid(axis="y", alpha=0.3)

        unique_pred, counts_pred = np.unique(self.all_predictions, return_counts=True)
        ax2.bar(
            range(len(unique_pred)),
            counts_pred,
            color="coral",
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title("Predicted Label Distribution", fontsize=14, fontweight="bold")
        ax2.set_xticks(range(len(unique_pred)))
        ax2.set_xticklabels(
            [self.id_to_label.get(i, f"Class_{i}") for i in unique_pred],
            rotation=45,
            ha="right",
        )
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Class Distribution Comparison - {self.run_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "class_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_attention_failures(self, max_samples: int = 8):
        if not self.failure_cases:
            return

        samples_to_analyze = self.failure_cases[:max_samples]

        for idx, case in enumerate(samples_to_analyze):
            try:
                batch_idx = case["idx"]
                batch = self.test_loader.dataset[batch_idx]

                pixel_values = batch["image"].unsqueeze(0).to(self.device)
                input_ids = batch["input_ids"].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    attn_map = self.model.get_cross_attention_map(pixel_values, input_ids)

                if attn_map is not None:
                    last_token_map = attn_map[0, -1, :, :]
                    save_path = self.output_dir / f"attention_failure_{idx}.png"

                    pred_label = self.id_to_label.get(
                        case["pred"], f"Class_{case['pred']}"
                    )
                    true_label = self.id_to_label.get(
                        case["label"], f"Class_{case['label']}"
                    )

                    title = f"Failure #{idx}: True={true_label}, Pred={pred_label}"
                    overlay_attention_heatmap(
                        pixel_values[0],
                        last_token_map,
                        title=title,
                        save_path=str(save_path),
                    )
            except Exception as e:
                print(f"  ⚠ Could not generate attention for failure {idx}: {e}")

    def _plot_confidence_vs_accuracy(self):
        confidences = np.array(
            [case["confidence"] for case in self.success_cases + self.failure_cases]
        )
        correct = np.array([1] * len(self.success_cases) + [0] * len(self.failure_cases))

        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        counts = []

        for thresh in thresholds:
            mask = confidences >= thresh
            if mask.sum() > 0:
                acc = (correct[mask].sum() / mask.sum()) * 100
                accuracies.append(acc)
                counts.append(int(mask.sum()))
            else:
                accuracies.append(0)
                counts.append(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(thresholds, accuracies, linewidth=2, color="darkblue")
        ax1.fill_between(thresholds, accuracies, alpha=0.3)
        ax1.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax1.set_title("Accuracy vs Confidence Threshold", fontsize=14, fontweight="bold")
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 105)

        ax2.plot(thresholds, counts, linewidth=2, color="darkgreen")
        ax2.fill_between(thresholds, counts, alpha=0.3, color="green")
        ax2.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Sample Count", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Samples Remaining at Confidence Threshold", fontsize=14, fontweight="bold"
        )
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 1)

        plt.suptitle(
            f"Confidence Analysis - {self.run_name}", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confidence_vs_accuracy.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def save_metrics_report(self, metrics: Dict):
        report_path = self.output_dir / "metrics_report.txt"
        json_path = self.output_dir / "metrics.json"

        unique_labels = [int(x) for x in metrics["unique_labels"]]

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Test Split: {self.test_split}\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy:           {metrics['accuracy']:.2f}%\n")
            f.write(f"Precision (Macro):  {metrics['precision_macro']:.2f}%\n")
            f.write(f"Recall (Macro):     {metrics['recall_macro']:.2f}%\n")
            f.write(f"F1-Score (Macro):   {metrics['f1_macro']:.2f}%\n\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.2f}%\n")
            f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.2f}%\n")
            f.write(f"F1-Score (Weighted):  {metrics['f1_weighted']:.2f}%\n\n")

            f.write("PER-CLASS METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for i, label_id in enumerate(unique_labels):
                class_name = self.id_to_label.get(label_id, f"Class_{label_id}")
                prec = metrics["per_class"]["precision"][i]
                rec = metrics["per_class"]["recall"][i]
                f1 = metrics["per_class"]["f1"][i]
                sup = int(metrics["per_class"]["support"][i])

                f.write(
                    f"{class_name:<20} {prec:>10.2f}%  {rec:>10.2f}%  {f1:>10.2f}%  {sup:>10}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("TP/TN/FP/FN BREAKDOWN (One-vs-Rest)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10}\n")
            f.write("-" * 80 + "\n")

            for i, label_id in enumerate(unique_labels):
                class_name = self.id_to_label.get(label_id, f"Class_{label_id}")
                tp = int(metrics["per_class"]["tp"][i])
                tn = int(metrics["per_class"]["tn"][i])
                fp = int(metrics["per_class"]["fp"][i])
                fn = int(metrics["per_class"]["fn"][i])

                f.write(f"{class_name:<20} {tp:<10} {tn:<10} {fp:<10} {fn:<10}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Total Samples:     {len(self.all_predictions)}\n")
            f.write(
                f"Correct:           {len(self.success_cases)} ({len(self.success_cases) / len(self.all_predictions) * 100:.2f}%)\n"
            )
            f.write(
                f"Incorrect:         {len(self.failure_cases)} ({len(self.failure_cases) / len(self.all_predictions) * 100:.2f}%)\n"
            )
            f.write("=" * 80 + "\n")

        print(f"  ✓ Saved metrics report to {report_path}")

        json_metrics = {
            "accuracy": float(metrics["accuracy"]),
            "precision_macro": float(metrics["precision_macro"]),
            "recall_macro": float(metrics["recall_macro"]),
            "f1_macro": float(metrics["f1_macro"]),
            "precision_weighted": float(metrics["precision_weighted"]),
            "recall_weighted": float(metrics["recall_weighted"]),
            "f1_weighted": float(metrics["f1_weighted"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "unique_labels": unique_labels,
            "per_class": {
                "precision": metrics["per_class"]["precision"].tolist(),
                "recall": metrics["per_class"]["recall"].tolist(),
                "f1": metrics["per_class"]["f1"].tolist(),
                "support": metrics["per_class"]["support"].tolist(),
                "tp": metrics["per_class"]["tp"].tolist(),
                "tn": metrics["per_class"]["tn"].tolist(),
                "fp": metrics["per_class"]["fp"].tolist(),
                "fn": metrics["per_class"]["fn"].tolist(),
            },
            "total_samples": len(self.all_predictions),
            "correct": len(self.success_cases),
            "incorrect": len(self.failure_cases),
        }

        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=2)

        print(f"  ✓ Saved metrics JSON to {json_path}")

    def run_full_analysis(self):
        print("\n" + "=" * 80)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("=" * 80)

        self.run_inference(max_samples=176810)
        metrics = self.compute_metrics()
        self.generate_visualizations(metrics)
        self.save_metrics_report(metrics)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def analyze_model(
    checkpoint_path: str,
    run_name: str,
    test_split: str = "test",
    custom_dataloader_fn=None,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    analyzer = ComprehensiveModelAnalyzer(
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        test_split=test_split,
        custom_dataloader_fn=custom_dataloader_fn,
    )

    analyzer.run_full_analysis()


if __name__ == "__main__":
    import re

    RUN_NAME = "traffic_vlm_v2_multi_class_1"
    CHECKPOINT_PATH = f"checkpoints/{RUN_NAME}/best_model.pt"
    TEST_SPLIT = "test"

    run_dir = Path("outputs") / RUN_NAME

    spec = spec_from_file_location("saved_dataset_config", run_dir / "dataset_config.py")
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load dataset config")
    dataset_module = module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    saved_cfg = dataset_module.DatasetConfig()

    DATASET_DIR = "data/processed_3"
    VOCAB_PATH = Path(DATASET_DIR) / "vocab.json"

    print("\n🔧 Using Run Configuration:")
    print(f"   Dataset Dir: {DATASET_DIR}")
    print(f"   Vocab Path: {VOCAB_PATH}")
    print(f"   Image Size: {saved_cfg.image_size}")
    print(f"   Num Classes: {saved_cfg.num_classes}\n")

    class CustomTokenizer:
        def __init__(self, vocab_path):
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"✓ Loaded {len(self.vocab)} tokens from {vocab_path}")

        def encode(self, text, max_len=None):
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            words = text.split()
            ids = [self.vocab.get(w, self.vocab.get("[UNK]", 3)) for w in words]
            ids = [self.vocab.get("[SOS]", 1)] + ids + [self.vocab.get("[EOS]", 2)]

            if max_len:
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    ids[-1] = self.vocab.get("[EOS]", 2)
                else:
                    ids = ids + [self.vocab.get("[PAD]", 0)] * (max_len - len(ids))
            return ids

    class CustomDataset(Dataset):
        def __init__(self, dataset_dir, split, cfg):
            self.dataset_dir = Path(dataset_dir)
            self.cfg = cfg
            self.split = split

            cmd_path = self.dataset_dir / f"{split}_commands.json"
            with open(cmd_path, "r") as f:
                self.commands = json.load(f)

            self.tokenizer = CustomTokenizer(self.dataset_dir / "vocab.json")

            if split == "train":
                self.transform = T.Compose(
                    [
                        T.Resize((cfg.image_size, cfg.image_size)),
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                        ),
                        T.RandomGrayscale(p=0.1),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=cfg.mean, std=cfg.std),
                    ]
                )
            else:
                self.transform = T.Compose(
                    [
                        T.Resize((cfg.image_size, cfg.image_size)),
                        T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=cfg.mean, std=cfg.std),
                    ]
                )

            self.h5_file = None
            self.images = None
            self.h5_path = self.dataset_dir / f"{split}.h5"

        def __len__(self):
            return len(self.commands)

        def __getitem__(self, idx):
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, "r")
                self.images = self.h5_file["images"]

            cmd = self.commands[idx]
            image_idx = cmd["image_idx"]

            img_raw = torch.from_numpy(self.images[image_idx]).permute(2, 0, 1)  # type: ignore
            image = self.transform(img_raw)

            input_ids = self.tokenizer.encode(cmd["q"], max_len=self.cfg.max_seq_len)

            a_text = cmd["a"].lower().strip()
            label_id = self.cfg.label_map.get(a_text, 1)

            return {
                "image": image,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label": torch.tensor(label_id, dtype=torch.long),
            }

    def custom_get_dataloader(split_name, batch_size=32, num_workers=4, shuffle=True):
        dataset = CustomDataset(DATASET_DIR, split_name, saved_cfg)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    analyze_model(
        checkpoint_path=CHECKPOINT_PATH,
        run_name=RUN_NAME,
        test_split=TEST_SPLIT,
        custom_dataloader_fn=custom_get_dataloader,
    )
