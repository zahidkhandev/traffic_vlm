import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def load_and_visualize_run(run_dir: str):
    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Loading results from: {run_dir}")
    print(f"{'=' * 60}\n")

    stats_file = os.path.join(run_dir, "stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)

        print("STATISTICS:")
        print(f"  Total Samples: {stats['total_samples']}")
        print(f"  Label Errors: {stats['label_errors']}")
        print(f"  Error Rate: {stats['error_rate']:.2%}")
        print("\n  Confidence Thresholds:")
        for cls, threshold in stats["thresholds"].items():
            print(f"    {cls:15s}: {threshold:.3f}")

    object_classes = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]

    confident_joint_file = os.path.join(run_dir, "confident_joint.npy")
    if os.path.exists(confident_joint_file):
        confident_joint = np.load(confident_joint_file)
        print(f"\nLoaded confident_joint.npy: shape {confident_joint.shape}")
        print("\nConfident Joint Matrix C(y_noisy, y_pred):")
        print(confident_joint)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            confident_joint,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=object_classes,
            yticklabels=object_classes,
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_title("Confident Joint Matrix", fontsize=16, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Given Label")
        plt.tight_layout()
        plt.show()

    noise_matrix_file = os.path.join(run_dir, "noise_matrix.npy")
    if os.path.exists(noise_matrix_file):
        noise_matrix = np.load(noise_matrix_file)
        print(f"\nLoaded noise_matrix.npy: shape {noise_matrix.shape}")
        print("\nNoise Matrix P(y_noisy | y_true):")
        print(noise_matrix)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(
            noise_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=object_classes,
            yticklabels=object_classes,
            ax=axes[0],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
        )
        axes[0].set_title("P(y_noisy | y_true)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("True Label")
        axes[0].set_ylabel("Noisy Label")

        sns.heatmap(
            noise_matrix.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=object_classes,
            yticklabels=object_classes,
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title("P(y_true | y_noisy)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Noisy Label")
        axes[1].set_ylabel("True Label")

        plt.tight_layout()
        plt.show()

    embeddings_file = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
        print(f"\nLoaded embeddings.npy: shape {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Number of samples: {embeddings.shape[0]}")

    error_crops_dir = os.path.join(run_dir, "label_errors_to_verify")
    if os.path.exists(error_crops_dir):
        error_files = [f for f in os.listdir(error_crops_dir) if f.endswith(".jpg")]
        print(f"\nLABEL ERRORS TO VERIFY: {len(error_files)} crops")
        print(f"  Location: {error_crops_dir}")

        if error_files:
            print("\n  Sample errors:")
            for error_file in sorted(error_files)[:5]:
                print(f"    {error_file}")
            if len(error_files) > 5:
                print(f"    ... and {len(error_files) - 5} more")

    print(f"\n{'=' * 60}\n")


def interactive_viewer():
    base_dir = "data/processed/confident_learning"

    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return

    runs = sorted(
        [
            d
            for d in os.listdir(base_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
        ]
    )

    if not runs:
        print(f"No runs found in {base_dir}")
        return

    print("\nAVAILABLE RUNS:")
    for idx, run in enumerate(runs, 1):
        run_path = os.path.join(base_dir, run)
        stats_file = os.path.join(run_path, "stats.json")

        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
            print(
                f"  {idx}. {run} - {stats['total_samples']} samples, {stats['label_errors']} errors ({stats['error_rate']:.1%})"
            )
        else:
            print(f"  {idx}. {run}")

    print("\n  0. Exit")

    try:
        choice = input("\nSelect run number (or 'latest'): ").strip()

        if choice == "0":
            return
        elif choice.lower() == "latest":
            selected_run = runs[-1]
        else:
            selected_run = runs[int(choice) - 1]

        run_path = os.path.join(base_dir, selected_run)
        load_and_visualize_run(run_path)

    except (ValueError, IndexError):
        print("Invalid selection")


if __name__ == "__main__":
    print("Confident Learning Results Viewer\n")

    if len(sys.argv) > 1:
        run_path = sys.argv[1]

        if not run_path.startswith("data/processed/confident_learning/run_"):
            run_path = os.path.join("data/processed/confident_learning", run_path)

        load_and_visualize_run(run_path)
    else:
        interactive_viewer()
