# Task 25: Self-Attention Plots
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_attention_matrix(
    attention_scores, tokens=None, title="Attention Matrix", save_path=None
):
    """
    Plots a raw NxN attention matrix (Self-Attention).

    Args:
        attention_scores: [Seq_Len, Seq_Len] tensor
        tokens: List of strings labels for axes (optional)
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_scores,
        xticklabels=tokens if tokens else "auto",
        yticklabels=tokens if tokens else "auto",
        cmap="viridis",
    )
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
