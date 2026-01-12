import argparse
import os
import re
import sys

import matplotlib.pyplot as plt


def get_unique_filename(filepath):
    if not os.path.exists(filepath):
        return filepath

    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    counter = 1

    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1


def parse_log_file(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None

    data = {}
    pattern = re.compile(
        r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)%"
    )

    print(f"Scanning {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                data[epoch] = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }

    if not data:
        print("No valid training data found in log file.")
        return None

    sorted_epochs = sorted(data.keys())
    train_losses = [data[e]["train_loss"] for e in sorted_epochs]
    val_losses = [data[e]["val_loss"] for e in sorted_epochs]
    val_accs = [data[e]["val_acc"] for e in sorted_epochs]

    return sorted_epochs, train_losses, val_losses, val_accs


def plot_curves(epochs, train_losses, val_losses, val_accs, output_path):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        epochs, train_losses, label="Train Loss", color="cornflowerblue", linewidth=2
    )
    plt.plot(epochs, val_losses, label="Val Loss", color="darkorange", linewidth=2)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        val_accs,
        label="Val Accuracy",
        color="forestgreen",
        marker="o",
        markersize=4,
        linewidth=2,
    )

    max_acc = max(val_accs)
    max_epoch = epochs[val_accs.index(max_acc)]
    plt.annotate(
        f"Max: {max_acc:.2f}%",
        xy=(max_epoch, max_acc),
        xytext=(max_epoch, max_acc + 5),
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
    )

    plt.title("Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully saved plot to {output_path}")
    print(f"Parsed {len(epochs)} unique epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="The folder name in outputs/")
    args = parser.parse_args()

    log_path = os.path.join("outputs", args.run_name, "train.log")

    base_output_img = os.path.join("outputs", args.run_name, "training_curve.png")
    output_img = get_unique_filename(base_output_img)

    result = parse_log_file(log_path)

    if result:
        epochs, t_loss, v_loss, v_acc = result
        plot_curves(epochs, t_loss, v_loss, v_acc, output_img)
        plt.show()
    else:
        sys.exit(1)
