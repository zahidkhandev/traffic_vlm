import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput


class QwenConfidentLearning:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.categories = [
            "pedestrian",
            "person",
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

    def get_class_probabilities(self, image_crop):
        probs = {}

        for category in self.categories:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_crop},
                        {
                            "type": "text",
                            "text": f"Is this a {category}? Answer yes or no.",
                        },
                    ],
                }
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )

            yes_token = self.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token = self.tokenizer.encode("no", add_special_tokens=False)[0]

            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            if (
                isinstance(outputs, GenerateDecoderOnlyOutput)
                and outputs.scores
                and len(outputs.scores) > 0
            ):
                first_token_logits = outputs.scores[0][0]
                yes_logit = first_token_logits[yes_token].item()
                no_logit = first_token_logits[no_token].item()
                yes_prob = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[
                    0
                ].item()
            else:
                yes_prob = 0.5

            probs[category] = yes_prob

        probs_array = np.array([probs[cat] for cat in self.categories])
        probs_normalized = probs_array / (probs_array.sum() + 1e-8)

        return {cat: float(prob) for cat, prob in zip(self.categories, probs_normalized)}


class BDD100KConfidentLearning:
    def __init__(self, model_path):
        self.vlm = QwenConfidentLearning(model_path)
        self.categories = self.vlm.categories
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.samples = []

    def process_dataset(self, images_path, labels_path, max_samples=None):
        label_files = [f for f in os.listdir(labels_path) if f.endswith(".json")]

        total = min(len(label_files), max_samples) if max_samples else len(label_files)

        for label_file in tqdm(label_files[:total], desc="Processing images"):
            file_id = os.path.splitext(label_file)[0]
            img_path = os.path.join(images_path, f"{file_id}.jpg")
            label_path = os.path.join(labels_path, label_file)

            if not os.path.exists(img_path):
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            full_img = Image.open(img_path)

            for frame in label_data.get("frames", []):
                for obj in frame.get("objects", []):
                    category = obj.get("category")
                    print(f"Found category: {category}")
                    if category not in self.categories:
                        print(f"Skipping unknown category: {category}")
                        continue

                    box = obj.get("box2d")
                    if not box:
                        continue

                    x1, y1 = int(box["x1"]), int(box["y1"])
                    x2, y2 = int(box["x2"]), int(box["y2"])

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = full_img.crop((x1, y1, x2, y2))

                    pred_probs = self.vlm.get_class_probabilities(crop)

                    self.samples.append(
                        {
                            "image": f"{file_id}.jpg",
                            "obj_id": obj.get("id"),
                            "given_label": category,
                            "given_label_idx": self.cat_to_idx[category],
                            "box": [x1, y1, x2, y2],
                            "pred_probs": pred_probs,
                        }
                    )

        print(f"\nTotal samples collected: {len(self.samples)}")

    def compute_confident_joint(self):
        if not self.samples:
            print("No samples to compute confident joint")
            return np.zeros((len(self.categories), len(self.categories)))

        n = len(self.categories)
        confident_joint = np.zeros((n, n))

        pred_probs_matrix = np.array(
            [[s["pred_probs"][cat] for cat in self.categories] for s in self.samples]
        )

        thresholds = np.mean(pred_probs_matrix, axis=0)

        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cat] for cat in self.categories])

            for pred_idx, (cat, prob) in enumerate(zip(self.categories, probs)):
                if prob >= thresholds[pred_idx]:
                    confident_joint[given_idx, pred_idx] += 1

        self.confident_joint = confident_joint
        self.thresholds = thresholds
        self.pred_probs_matrix = pred_probs_matrix
        return confident_joint

    def estimate_latent_joint(self):
        Q_joint = self.confident_joint / (self.confident_joint.sum() + 1e-8)
        return Q_joint

    def estimate_noise_matrix(self):
        Q_joint = self.estimate_latent_joint()
        py = Q_joint.sum(axis=0)
        noise_matrix = Q_joint / (py + 1e-8)
        return noise_matrix

    def compute_label_quality_scores(self):
        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cat] for cat in self.categories])

            pred_idx = np.argmax(probs)

            sample["self_confidence"] = float(probs[given_idx])
            sample["predicted_label_idx"] = pred_idx
            sample["predicted_label"] = self.categories[pred_idx]
            sample["predicted_confidence"] = float(probs[pred_idx])

            sorted_probs = np.sort(probs)[::-1]
            sample["margin"] = float(sorted_probs[0] - sorted_probs[1])

            sample["normalized_margin"] = float(
                sample["margin"] / (sorted_probs[0] + 1e-8)
            )

    def find_label_errors(self):
        errors = []

        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            pred_idx = sample["predicted_label_idx"]

            if (
                pred_idx != given_idx
                and sample["predicted_confidence"] >= self.thresholds[pred_idx]
            ):
                sample["is_error"] = True
                errors.append(sample.copy())
            else:
                sample["is_error"] = False

        return errors

    def visualize_all(self, images_path, output_dir):
        if not self.samples:
            print("No samples to visualize")
            return

        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        self._visualize_confident_joint(viz_dir)
        self._visualize_noise_matrices(viz_dir)
        self._visualize_probability_distributions(viz_dir)
        self._visualize_embeddings(viz_dir)
        self._visualize_label_quality_scores(viz_dir)
        self._visualize_error_examples(images_path, viz_dir)

    def _visualize_confident_joint(self, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            self.confident_joint,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[0],
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title("Confident Joint Matrix (Raw Counts)", fontsize=14)
        axes[0].set_xlabel("Predicted Label (ŷ*)")
        axes[0].set_ylabel("Given Label (ỹ)")

        Q_joint = self.estimate_latent_joint()
        sns.heatmap(
            Q_joint,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[1],
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title("Normalized Confident Joint (Q̃)", fontsize=14)
        axes[1].set_xlabel("Predicted Label (ŷ*)")
        axes[1].set_ylabel("Given Label (ỹ)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confident_joint.png"), dpi=300)
        plt.close()

    def _visualize_noise_matrices(self, output_dir):
        noise_matrix = self.estimate_noise_matrix()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            noise_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[0],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "P(ỹ|y*)"},
        )
        axes[0].set_title("Noise Matrix P(ỹ|y*)", fontsize=14)
        axes[0].set_xlabel("True Label (y*)")
        axes[0].set_ylabel("Given Label (ỹ)")

        inverse_noise = noise_matrix.T
        sns.heatmap(
            inverse_noise,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "P(y*|ỹ)"},
        )
        axes[1].set_title("Inverse Noise Matrix P(y*|ỹ)", fontsize=14)
        axes[1].set_xlabel("Given Label (ỹ)")
        axes[1].set_ylabel("True Label (y*)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "noise_matrices.png"), dpi=300)
        plt.close()

    def _visualize_probability_distributions(self, output_dir):
        if not self.samples:
            print("No samples for probability distributions")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self_conf = [s.get("self_confidence", 0.5) for s in self.samples]

        if not self_conf or all(np.isnan(self_conf)):
            print("No valid confidence scores")
            plt.close()
            return

        axes[0, 0].hist(self_conf, bins=30, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Self-Confidence Distribution")
        axes[0, 0].set_xlabel("P(ỹ|x)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            np.mean(self_conf),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(self_conf):.3f}",
        )
        axes[0, 0].legend()

        margins = [s.get("margin", 0) for s in self.samples]
        axes[0, 1].hist(margins, bins=30, color="lightgreen", edgecolor="black")
        axes[0, 1].set_title("Margin Distribution")
        axes[0, 1].set_xlabel("Margin (P₁ - P₂)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            np.mean(margins),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(margins):.3f}",
        )
        axes[0, 1].legend()

        errors = [s for s in self.samples if s.get("is_error", False)]
        correct = [s for s in self.samples if not s.get("is_error", False)]

        if errors and correct:
            axes[1, 0].hist(
                [s["self_confidence"] for s in correct],
                bins=20,
                alpha=0.5,
                label="Correct",
                color="green",
            )
            axes[1, 0].hist(
                [s["self_confidence"] for s in errors],
                bins=20,
                alpha=0.5,
                label="Errors",
                color="red",
            )
            axes[1, 0].set_title("Self-Confidence: Correct vs Errors")
            axes[1, 0].set_xlabel("Self-Confidence")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, "No errors found", ha="center", va="center")
            axes[1, 0].set_title("Self-Confidence: Correct vs Errors")

        threshold_line = list(self.thresholds)
        axes[1, 1].bar(range(len(self.categories)), threshold_line, color="coral")
        axes[1, 1].set_xticks(range(len(self.categories)))
        axes[1, 1].set_xticklabels(self.categories, rotation=45, ha="right")
        axes[1, 1].set_title("Thresholds per Class")
        axes[1, 1].set_ylabel("Threshold τⱼ")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "probability_distributions.png"), dpi=300)
        plt.close()

    def _visualize_embeddings(self, output_dir):
        if len(self.samples) < 10:
            return

        X = self.pred_probs_matrix
        y_given = np.array([s["given_label_idx"] for s in self.samples])
        y_pred = np.array([s["predicted_label_idx"] for s in self.samples])
        is_error = np.array([s.get("is_error", False) for s in self.samples])

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        if len(X) >= 50:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
            X_tsne = tsne.fit_transform(X)

            scatter = axes[0, 0].scatter(
                X_tsne[:, 0], X_tsne[:, 1], c=y_given, cmap="tab10", alpha=0.6, s=50
            )
            axes[0, 0].set_title("t-SNE: Colored by Given Label")
            plt.colorbar(scatter, ax=axes[0, 0])

            colors = ["green" if not e else "red" for e in is_error]
            axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=50)
            axes[0, 1].set_title("t-SNE: Green=Correct, Red=Error")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = axes[1, 0].scatter(
            X_pca[:, 0], X_pca[:, 1], c=y_given, cmap="tab10", alpha=0.6, s=50
        )
        axes[1, 0].set_title(
            f"PCA: Colored by Given Label\nExplained Variance: {sum(pca.explained_variance_ratio_):.2%}"
        )
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        plt.colorbar(scatter, ax=axes[1, 0])

        colors = ["green" if not e else "red" for e in is_error]
        axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
        axes[1, 1].set_title("PCA: Green=Correct, Red=Error")
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "embeddings_visualization.png"), dpi=300)
        plt.close()

    def _visualize_label_quality_scores(self, output_dir):
        df = pd.DataFrame(self.samples)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        df_sorted = df.sort_values("self_confidence")
        axes[0, 0].scatter(
            range(len(df_sorted)),
            df_sorted["self_confidence"],
            c=df_sorted["is_error"].astype(int),
            cmap="RdYlGn_r",
            alpha=0.6,
        )
        axes[0, 0].set_title("Label Quality Score (Self-Confidence)")
        axes[0, 0].set_xlabel("Sample Index (sorted)")
        axes[0, 0].set_ylabel("Self-Confidence")
        axes[0, 0].axhline(
            df["self_confidence"].median(), color="blue", linestyle="--", label="Median"
        )
        axes[0, 0].legend()

        df_sorted_margin = df.sort_values("margin")
        axes[0, 1].scatter(
            range(len(df_sorted_margin)),
            df_sorted_margin["margin"],
            c=df_sorted_margin["is_error"].astype(int),
            cmap="RdYlGn_r",
            alpha=0.6,
        )
        axes[0, 1].set_title("Margin Score")
        axes[0, 1].set_xlabel("Sample Index (sorted)")
        axes[0, 1].set_ylabel("Margin")

        category_error_rates = df.groupby("given_label")["is_error"].mean()
        axes[1, 0].bar(
            category_error_rates.index, category_error_rates.values, color="salmon"
        )
        axes[1, 0].set_xticklabels(category_error_rates.index, rotation=45, ha="right")
        axes[1, 0].set_title("Error Rate by Category")
        axes[1, 0].set_ylabel("Error Rate")

        category_counts = df["given_label"].value_counts()
        axes[1, 1].bar(category_counts.index, category_counts.values, color="steelblue")
        axes[1, 1].set_xticklabels(category_counts.index, rotation=45, ha="right")
        axes[1, 1].set_title("Sample Distribution by Category")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "label_quality_scores.png"), dpi=300)
        plt.close()

    def _visualize_error_examples(self, images_path, output_dir):
        errors = [s for s in self.samples if s.get("is_error", False)]

        if not errors:
            print("No errors to visualize")
            return

        errors_sorted = sorted(errors, key=lambda x: x["margin"])[:12]

        n_cols = 4
        n_rows = min(3, (len(errors_sorted) + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

        for idx, error in enumerate(errors_sorted[: len(axes)]):
            img_path = os.path.join(images_path, error["image"])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            crop = img.crop(error["box"])

            axes[idx].imshow(crop)
            axes[idx].axis("off")

            title = (
                f"Given: {error['given_label']}\n"
                f"Pred: {error['predicted_label']}\n"
                f"Conf: {error['predicted_confidence']:.2f}\n"
                f"Margin: {error['margin']:.2f}"
            )
            axes[idx].set_title(title, fontsize=9)

        for idx in range(len(errors_sorted), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_examples.png"), dpi=300)
        plt.close()

    def save_results(self, images_path, output_dir="data/processed/confident_learning"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        errors_dir = os.path.join(run_dir, "label_errors")
        os.makedirs(errors_dir, exist_ok=True)

        self.compute_label_quality_scores()
        errors = self.find_label_errors()

        for error in tqdm(errors[:200], desc="Saving error crops"):
            img_path = os.path.join(images_path, error["image"])
            if not os.path.exists(img_path):
                continue
            with Image.open(img_path) as img:
                crop = img.crop(error["box"])
                crop_name = f"{error['obj_id']}_given_{error['given_label']}_pred_{error['predicted_label']}.jpg"
                crop.save(os.path.join(errors_dir, crop_name))

        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(os.path.join(run_dir, "label_errors.csv"), index=False)

        all_df = pd.DataFrame(self.samples)
        all_df.to_csv(os.path.join(run_dir, "all_samples.csv"), index=False)

        np.save(os.path.join(run_dir, "confident_joint.npy"), self.confident_joint)

        noise_matrix = self.estimate_noise_matrix()
        np.save(os.path.join(run_dir, "noise_matrix.npy"), noise_matrix)

        stats = {
            "total_samples": len(self.samples),
            "label_errors": len(errors),
            "error_rate": (
                len(errors) / len(self.samples) if len(self.samples) > 0 else 0
            ),
            "thresholds": {
                cat: float(t) for cat, t in zip(self.categories, self.thresholds)
            },
        }

        with open(os.path.join(run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        self.visualize_all(images_path, run_dir)

        print(f"\nResults saved to {run_dir}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label errors found: {stats['label_errors']} ({stats['error_rate']:.2%})")

        return errors


model_path = r"C:\Users\nkz3kor\Documents\traffic_vlm\qwen-model"
cl = BDD100KConfidentLearning(model_path)

images_path = r"data\raw\mini\images\test"
labels_path = r"data\raw\mini\labels\test"

cl.process_dataset(images_path, labels_path, max_samples=12)
cl.compute_confident_joint()
errors = cl.save_results(images_path)
