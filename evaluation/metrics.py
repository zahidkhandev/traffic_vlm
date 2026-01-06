import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(predictions, labels):
    """
    Comprehensive Evaluation Metrics.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Confusion Matrix (Flattened: TN, FP, FN, TP)
    # Ensure labels=[0, 1] to handle cases where a batch might miss a class
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def print_metrics_summary(metrics):
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")
    print("-" * 20)
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print("      Pred NO   Pred YES")
    print(f"Act NO   {cm['TN']:<8} {cm['FP']:<8}")
    print(f"Act YES  {cm['FN']:<8} {cm['TP']:<8}")
    print("=" * 40 + "\n")
