import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_crop},
                    {
                        "type": "text",
                        "text": "Describe this object in one word. What is it?",
                    },
                ],
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        if hasattr(outputs, "sequences"):
            generated_text = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
        else:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"VLM says: {generated_text[-100:].strip()}")

        if outputs.scores and len(outputs.scores) > 0:
            first_token_logits = outputs.scores[0][0]

            logits_dict = {}
            for cat in self.categories:
                tokens = self.tokenizer.encode(cat, add_special_tokens=False)
                if tokens:
                    token_id = tokens[0]
                    logits_dict[cat] = first_token_logits[token_id].item()
                else:
                    logits_dict[cat] = -1000.0

            top3_logits = sorted(logits_dict.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            print(f"Top 3 logits: {[(k, f'{v:.1f}') for k, v in top3_logits]}")

            logits_array = torch.tensor([logits_dict[cat] for cat in self.categories])
            probs = torch.softmax(logits_array, dim=0).numpy()

            top3_probs = [
                (self.categories[i], probs[i]) for i in np.argsort(probs)[::-1][:3]
            ]
            print(f"Top 3 probs: {[(k, f'{v:.3f}') for k, v in top3_probs]}")
            print(f"Prob std: {probs.std():.4f}\n")

            return {cat: float(prob) for cat, prob in zip(self.categories, probs)}
        else:
            return {cat: 1.0 / len(self.categories) for cat in self.categories}


class VoxelVisualizer:
    def __init__(self, voxel_resolution=(50, 50, 20)):
        self.voxel_resolution = voxel_resolution
        self.voxel_grid = None

    def create_voxel_grid_from_detections(
        self, samples, image_width=1280, image_height=720
    ):
        voxel_grid = np.zeros(self.voxel_resolution)
        category_grid = np.zeros(self.voxel_resolution, dtype=int)
        confidence_grid = np.zeros(self.voxel_resolution)

        for sample in samples:
            box = sample["box"]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            confidence = sample.get("self_confidence", 0.5)
            if np.isnan(confidence) or confidence is None:
                confidence = 0.5

            vx = int((x_center / image_width) * self.voxel_resolution[0])
            vy = int((y_center / image_height) * self.voxel_resolution[1])
            vz = int(confidence * self.voxel_resolution[2])

            vx = np.clip(vx, 0, self.voxel_resolution[0] - 1)
            vy = np.clip(vy, 0, self.voxel_resolution[1] - 1)
            vz = np.clip(vz, 0, self.voxel_resolution[2] - 1)

            voxel_grid[vx, vy, vz] = 1
            category_grid[vx, vy, vz] = sample["given_label_idx"]
            confidence_grid[vx, vy, vz] = confidence

        self.voxel_grid = voxel_grid
        self.category_grid = category_grid
        self.confidence_grid = confidence_grid
        return voxel_grid, category_grid, confidence_grid

    def visualize_voxel_space(self, samples, categories, output_path):
        voxel_grid, category_grid, confidence_grid = (
            self.create_voxel_grid_from_detections(samples)
        )
        occupied = np.where(voxel_grid == 1)
        if len(occupied[0]) == 0:
            print("No voxels to visualize")
            return

        x_coords = occupied[0]
        y_coords = occupied[1]
        z_coords = occupied[2]
        colors = [
            categories[category_grid[x, y, z]]
            for x, y, z in zip(x_coords, y_coords, z_coords)
        ]
        confidences = [
            confidence_grid[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)
        ]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=confidences,
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title="Confidence"),
                        line=dict(width=0.5, color="white"),
                    ),
                    text=[
                        f"Category: {c}<br>Confidence: {conf:.3f}"
                        for c, conf in zip(colors, confidences)
                    ],
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title="3D Voxel Space: Traffic Objects (Z-axis = Confidence)",
            scene=dict(
                xaxis_title="X (Image Width)",
                yaxis_title="Y (Image Height)",
                zaxis_title="Z (Confidence Level)",
                bgcolor="rgba(240,240,240,0.9)",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(output_path)
        print(f"Saved 3D voxel visualization to {output_path}")

    def visualize_category_voxels(self, samples, categories, output_path):
        voxel_grid, category_grid, confidence_grid = (
            self.create_voxel_grid_from_detections(samples)
        )
        occupied = np.where(voxel_grid == 1)
        if len(occupied[0]) == 0:
            print("No voxels to visualize")
            return

        x_coords = occupied[0]
        y_coords = occupied[1]
        z_coords = occupied[2]
        category_colors = [
            category_grid[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)
        ]
        category_names = [categories[cat] for cat in category_colors]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=category_colors,
                        colorscale="Rainbow",
                        opacity=0.7,
                        showscale=False,
                        line=dict(width=0.5, color="black"),
                    ),
                    text=[f"Category: {name}" for name in category_names],
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title="3D Voxel Space: Colored by Object Category",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Confidence",
                bgcolor="rgba(240,240,240,0.9)",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(output_path)
        print(f"Saved category voxel visualization to {output_path}")

    def visualize_error_voxels(self, samples, output_path):
        errors = [s for s in samples if s.get("is_error", False)]
        correct = [s for s in samples if not s.get("is_error", False)]
        fig = go.Figure()

        if correct:
            correct_data = self._extract_voxel_coords(correct)
            fig.add_trace(
                go.Scatter3d(
                    x=correct_data["x"],
                    y=correct_data["y"],
                    z=correct_data["z"],
                    mode="markers",
                    name="Correct",
                    marker=dict(size=6, color="green", opacity=0.6),
                )
            )

        if errors:
            error_data = self._extract_voxel_coords(errors)
            fig.add_trace(
                go.Scatter3d(
                    x=error_data["x"],
                    y=error_data["y"],
                    z=error_data["z"],
                    mode="markers",
                    name="Label Errors",
                    marker=dict(size=8, color="red", opacity=0.8, symbol="diamond"),
                )
            )

        fig.update_layout(
            title="3D Voxel Space: Label Quality (Green=Correct, Red=Error)",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Confidence",
                bgcolor="rgba(240,240,240,0.9)",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(output_path)
        print(f"Saved error voxel visualization to {output_path}")

    def _extract_voxel_coords(self, samples, image_width=1280, image_height=720):
        x_coords, y_coords, z_coords = [], [], []
        for sample in samples:
            box = sample["box"]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            confidence = sample.get("self_confidence", 0.5)
            if np.isnan(confidence) or confidence is None:
                confidence = 0.5

            vx = (x_center / image_width) * self.voxel_resolution[0]
            vy = (y_center / image_height) * self.voxel_resolution[1]
            vz = confidence * self.voxel_resolution[2]
            x_coords.append(vx)
            y_coords.append(vy)
            z_coords.append(vz)
        return {"x": x_coords, "y": y_coords, "z": z_coords}

    def visualize_density_heatmap(self, samples, output_path):
        voxel_grid, _, _ = self.create_voxel_grid_from_detections(samples)
        density_map = np.sum(voxel_grid, axis=2)
        fig = go.Figure(
            data=go.Heatmap(
                z=density_map.T, colorscale="Hot", colorbar=dict(title="Object Density")
            )
        )
        fig.update_layout(
            title="Object Density Heatmap (XY Projection)",
            xaxis_title="X (Image Width)",
            yaxis_title="Y (Image Height)",
            width=900,
            height=700,
        )
        fig.write_html(output_path)
        print(f"Saved density heatmap to {output_path}")


class BDD100KConfidentLearning:
    def __init__(self, model_path):
        self.vlm = QwenConfidentLearning(model_path)
        self.categories = self.vlm.categories
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.samples = []
        self.voxel_viz = VoxelVisualizer(voxel_resolution=(50, 50, 20))

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
                    if category not in self.categories:
                        continue
                    box = obj.get("box2d")
                    if not box:
                        continue
                    x1, y1 = int(box["x1"]), int(box["y1"])
                    x2, y2 = int(box["x2"]), int(box["y2"])
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = full_img.crop((x1, y1, x2, y2))
                    print(f"\n{'=' * 60}")
                    print(f"Ground truth: {category}")
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

        print(f"\n{'=' * 60}")
        print("PROBABILITY MATRIX DIAGNOSTICS:")
        print(f"Shape: {pred_probs_matrix.shape}")
        print(f"Overall std: {pred_probs_matrix.std():.4f} (should be > 0.05)")
        print(f"Mean: {pred_probs_matrix.mean():.4f}")

        thresholds = np.mean(pred_probs_matrix, axis=0)
        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cat] for cat in self.categories])
            for pred_idx, (_cat, prob) in enumerate(zip(self.categories, probs)):
                if prob >= thresholds[pred_idx]:
                    confident_joint[given_idx, pred_idx] += 1

        print(f"\nConfident Joint Diagonal sum: {np.diag(confident_joint).sum()}")
        print(
            f"Confident Joint Off-diagonal sum: {confident_joint.sum() - np.diag(confident_joint).sum()}"
        )

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
                print(
                    f"ERROR: Given={sample['given_label']}, Pred={sample['predicted_label']}, Conf={sample['predicted_confidence']:.3f}"
                )
            else:
                sample["is_error"] = False
        return errors

    def visualize_all(self, images_path, output_dir):
        if not self.samples:
            print("No samples to visualize")
            return

        viz_dir = os.path.join(output_dir, "visualizations")
        voxel_dir = os.path.join(output_dir, "voxel_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(voxel_dir, exist_ok=True)

        self._visualize_confident_joint(viz_dir)
        self._visualize_noise_matrices(viz_dir)
        self._visualize_probability_distributions(viz_dir)
        self._visualize_embeddings(viz_dir)
        self._visualize_label_quality_scores(viz_dir)
        self._visualize_error_examples(images_path, viz_dir)

        print("\n=== Generating 3D Voxel Visualizations ===")
        self.voxel_viz.visualize_voxel_space(
            self.samples, self.categories, os.path.join(voxel_dir, "voxel_space_3d.html")
        )
        self.voxel_viz.visualize_category_voxels(
            self.samples,
            self.categories,
            os.path.join(voxel_dir, "voxel_categories_3d.html"),
        )
        self.voxel_viz.visualize_error_voxels(
            self.samples, os.path.join(voxel_dir, "voxel_errors_3d.html")
        )
        self.voxel_viz.visualize_density_heatmap(
            self.samples, os.path.join(voxel_dir, "density_heatmap.html")
        )

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
            print("Skipping embeddings: less than 10 samples")
            return

        X = self.pred_probs_matrix.copy()
        if np.isnan(X).any():
            print("Warning: NaN values detected. Using imputer...")
            imputer = SimpleImputer(
                strategy="constant", fill_value=1.0 / len(self.categories)
            )
            X = imputer.fit_transform(X)

        if np.isnan(X).any() or np.allclose(X, X[0]):
            print("Skipping embeddings visualization due to data quality issues")
            return

        y_given = np.array([s["given_label_idx"] for s in self.samples])
        is_error = np.array([s.get("is_error", False) for s in self.samples])

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        if len(X) >= 50:
            try:
                tsne = TSNE(
                    n_components=2, random_state=42, perplexity=min(30, len(X) - 1)
                )
                X_tsne = tsne.fit_transform(X)
                scatter = axes[0, 0].scatter(
                    X_tsne[:, 0], X_tsne[:, 1], c=y_given, cmap="tab10", alpha=0.6, s=50
                )
                axes[0, 0].set_title("t-SNE: Colored by Given Label")
                plt.colorbar(scatter, ax=axes[0, 0])
                colors = ["green" if not e else "red" for e in is_error]
                axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=50)
                axes[0, 1].set_title("t-SNE: Green=Correct, Red=Error")
            except Exception as e:
                print(f"t-SNE failed: {e}")
                axes[0, 0].text(0.5, 0.5, "t-SNE failed", ha="center", va="center")
                axes[0, 1].text(0.5, 0.5, "t-SNE failed", ha="center", va="center")
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                f"t-SNE needs >= 50 samples (have {len(X)})",
                ha="center",
                va="center",
            )
            axes[0, 1].text(
                0.5,
                0.5,
                f"t-SNE needs >= 50 samples (have {len(X)})",
                ha="center",
                va="center",
            )

        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            scatter = axes[1, 0].scatter(
                X_pca[:, 0], X_pca[:, 1], c=y_given, cmap="tab10", alpha=0.6, s=50
            )
            axes[1, 0].set_title(
                f"PCA: Colored by Given Label\nVar: {sum(pca.explained_variance_ratio_):.2%}"
            )
            axes[1, 0].set_xlabel("PC1")
            axes[1, 0].set_ylabel("PC2")
            plt.colorbar(scatter, ax=axes[1, 0])
            colors = ["green" if not e else "red" for e in is_error]
            axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
            axes[1, 1].set_title("PCA: Green=Correct, Red=Error")
            axes[1, 1].set_xlabel("PC1")
            axes[1, 1].set_ylabel("PC2")
        except Exception as e:
            print(f"PCA failed: {e}")
            axes[1, 0].text(0.5, 0.5, "PCA failed", ha="center", va="center")
            axes[1, 1].text(0.5, 0.5, "PCA failed", ha="center", va="center")

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
            range(len(category_error_rates)), category_error_rates.values, color="salmon"
        )
        axes[1, 0].set_xticks(range(len(category_error_rates)))
        axes[1, 0].set_xticklabels(category_error_rates.index, rotation=45, ha="right")
        axes[1, 0].set_title("Error Rate by Category")
        axes[1, 0].set_ylabel("Error Rate")

        category_counts = df["given_label"].value_counts()
        axes[1, 1].bar(
            range(len(category_counts)), category_counts.values, color="steelblue"
        )
        axes[1, 1].set_xticks(range(len(category_counts)))
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
        if n_rows == 0:
            return

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
            title = f"Given: {error['given_label']}\nPred: {error['predicted_label']}\nConf: {error['predicted_confidence']:.2f}\nMargin: {error['margin']:.2f}"
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
            "error_rate": len(errors) / len(self.samples) if len(self.samples) > 0 else 0,
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


if __name__ == "__main__":
    model_path = r"C:\Users\nkz3kor\Documents\traffic_vlm\qwen-model"
    cl = BDD100KConfidentLearning(model_path)

    images_path = r"data\raw\mini-x\images\test"
    labels_path = r"data\raw\mini-x\labels\test"

    cl.process_dataset(images_path, labels_path, max_samples=3)
    cl.compute_confident_joint()
    errors = cl.save_results(images_path)
