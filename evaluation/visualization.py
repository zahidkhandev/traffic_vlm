import os

import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir="outputs"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val Loss")

    plt.title("Training Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()
