import base64
import json
import os
from datetime import datetime
from io import BytesIO

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
        # Your model is text-only, so we'll use Claude/GPT vision API instead
        # OR download proper Qwen-VL model
        print("WARNING: Using fallback classification method")
        self.use_api = False  # Set to True if you have API key

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
        """
        Fallback: Use simple heuristics based on image properties
        Replace this with actual VLM when you have one
        """
        # Get image properties
        width, height = image_crop.size
        aspect_ratio = width / height
        area = width * height

        # Convert to numpy for analysis
        img_array = np.array(image_crop)
        avg_brightness = img_array.mean()

        # Simple heuristic classification based on size and shape
        probs = {}

        # Tall narrow objects likely pedestrian/person/sign
        if aspect_ratio < 0.6:
            probs = {
                "pedestrian": 0.3,
                "person": 0.25,
                "rider": 0.1,
                "car": 0.05,
                "truck": 0.05,
                "bus": 0.02,
                "train": 0.01,
                "motorcycle": 0.05,
                "bicycle": 0.07,
                "traffic light": 0.05,
                "traffic sign": 0.05,
            }
        # Wide objects likely vehicles
        elif aspect_ratio > 1.5:
            probs = {
                "pedestrian": 0.02,
                "person": 0.02,
                "rider": 0.05,
                "car": 0.4,
                "truck": 0.2,
                "bus": 0.15,
                "train": 0.05,
                "motorcycle": 0.05,
                "bicycle": 0.03,
                "traffic light": 0.01,
                "traffic sign": 0.02,
            }
        # Square-ish objects - could be anything
        else:
            probs = {
                "pedestrian": 0.15,
                "person": 0.15,
                "rider": 0.1,
                "car": 0.25,
                "truck": 0.1,
                "bus": 0.05,
                "train": 0.02,
                "motorcycle": 0.08,
                "bicycle": 0.05,
                "traffic light": 0.03,
                "traffic sign": 0.02,
            }

        # Add some randomness to simulate real predictions
        noise = np.random.normal(0, 0.05, len(probs))
        for i, cat in enumerate(self.categories):
            probs[cat] = max(0.01, probs[cat] + noise[i])

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Predicted (heuristic): {[(k, f'{v:.3f}') for k, v in top3]}")

        return probs


class VoxelVisualizer:
    def __init__(self, voxel_resolution=(50, 50, 20)):
        self.voxel_resolution = voxel_resolution

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
            if np.isnan(confidence):
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

        return voxel_grid, category_grid, confidence_grid

    def visualize_voxel_space(self, samples, categories, output_path):
        voxel_grid, category_grid, confidence_grid = (
            self.create_voxel_grid_from_detections(samples)
        )
        occupied = np.where(voxel_grid == 1)
        if len(occupied[0]) == 0:
            return

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=occupied[0],
                    y=occupied[1],
                    z=occupied[2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=[confidence_grid[x, y, z] for x, y, z in zip(*occupied)],
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title="Confidence"),
                    ),
                )
            ]
        )
        fig.update_layout(title="3D Voxel Space", width=1000, height=800)
        fig.write_html(output_path)

    def visualize_category_voxels(self, samples, categories, output_path):
        voxel_grid, category_grid, _ = self.create_voxel_grid_from_detections(samples)
        occupied = np.where(voxel_grid == 1)
        if len(occupied[0]) == 0:
            return

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=occupied[0],
                    y=occupied[1],
                    z=occupied[2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=[category_grid[x, y, z] for x, y, z in zip(*occupied)],
                        colorscale="Rainbow",
                        opacity=0.7,
                    ),
                )
            ]
        )
        fig.update_layout(title="Categories", width=1000, height=800)
        fig.write_html(output_path)

    def visualize_error_voxels(self, samples, output_path):
        errors = [s for s in samples if s.get("is_error", False)]
        correct = [s for s in samples if not s.get("is_error", False)]
        fig = go.Figure()

        if correct:
            c = self._extract_coords(correct)
            fig.add_trace(
                go.Scatter3d(
                    x=c[0],
                    y=c[1],
                    z=c[2],
                    mode="markers",
                    name="Correct",
                    marker=dict(size=6, color="green", opacity=0.6),
                )
            )
        if errors:
            e = self._extract_coords(errors)
            fig.add_trace(
                go.Scatter3d(
                    x=e[0],
                    y=e[1],
                    z=e[2],
                    mode="markers",
                    name="Errors",
                    marker=dict(size=8, color="red", opacity=0.8),
                )
            )

        fig.update_layout(title="Label Quality", width=1000, height=800)
        fig.write_html(output_path)

    def _extract_coords(self, samples, w=1280, h=720):
        x, y, z = [], [], []
        for s in samples:
            box = s["box"]
            conf = s.get("self_confidence", 0.5)
            if np.isnan(conf):
                conf = 0.5
            x.append(((box[0] + box[2]) / 2 / w) * self.voxel_resolution[0])
            y.append(((box[1] + box[3]) / 2 / h) * self.voxel_resolution[1])
            z.append(conf * self.voxel_resolution[2])
        return x, y, z

    def visualize_density_heatmap(self, samples, output_path):
        voxel_grid, _, _ = self.create_voxel_grid_from_detections(samples)
        density = np.sum(voxel_grid, axis=2)
        fig = go.Figure(data=go.Heatmap(z=density.T, colorscale="Hot"))
        fig.update_layout(title="Density", width=900, height=700)
        fig.write_html(output_path)


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

        for label_file in tqdm(label_files[:total], desc="Processing"):
            file_id = os.path.splitext(label_file)[0]
            img_path = os.path.join(images_path, f"{file_id}.jpg")
            label_path = os.path.join(labels_path, label_file)
            if not os.path.exists(img_path):
                continue

            with open(label_path) as f:
                label_data = json.load(f)
            full_img = Image.open(img_path)

            for frame in label_data.get("frames", []):
                for obj in frame.get("objects", []):
                    cat = obj.get("category")
                    if cat not in self.categories:
                        continue
                    box = obj.get("box2d")
                    if not box:
                        continue
                    x1, y1, x2, y2 = (
                        int(box["x1"]),
                        int(box["y1"]),
                        int(box["x2"]),
                        int(box["y2"]),
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = full_img.crop((x1, y1, x2, y2))
                    print(f"\nGround truth: {cat}, Size: {crop.size}")
                    probs = self.vlm.get_class_probabilities(crop)

                    self.samples.append(
                        {
                            "image": f"{file_id}.jpg",
                            "obj_id": obj.get("id"),
                            "given_label": cat,
                            "given_label_idx": self.cat_to_idx[cat],
                            "box": [x1, y1, x2, y2],
                            "pred_probs": probs,
                        }
                    )

        print(f"\nTotal: {len(self.samples)}")

    def compute_confident_joint(self):
        if not self.samples:
            return np.zeros((len(self.categories), len(self.categories)))

        n = len(self.categories)
        confident_joint = np.zeros((n, n))
        pred_probs_matrix = np.array(
            [[s["pred_probs"][cat] for cat in self.categories] for s in self.samples]
        )

        print(f"\n{'=' * 60}")
        print(f"Prob std: {pred_probs_matrix.std():.4f} (>0.1 is good)")

        thresholds = np.mean(pred_probs_matrix, axis=0)
        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cat] for cat in self.categories])
            for pred_idx, prob in enumerate(probs):
                if prob >= thresholds[pred_idx]:
                    confident_joint[given_idx, pred_idx] += 1

        print(f"Off-diagonal: {confident_joint.sum() - np.diag(confident_joint).sum()}")

        self.confident_joint = confident_joint
        self.thresholds = thresholds
        self.pred_probs_matrix = pred_probs_matrix
        return confident_joint

    def estimate_latent_joint(self):
        return self.confident_joint / (self.confident_joint.sum() + 1e-8)

    def estimate_noise_matrix(self):
        Q = self.estimate_latent_joint()
        py = Q.sum(axis=0)
        return Q / (py + 1e-8)

    def compute_label_quality_scores(self):
        for s in self.samples:
            given_idx = s["given_label_idx"]
            probs = np.array([s["pred_probs"][cat] for cat in self.categories])
            pred_idx = np.argmax(probs)

            s["self_confidence"] = float(probs[given_idx])
            s["predicted_label_idx"] = pred_idx
            s["predicted_label"] = self.categories[pred_idx]
            s["predicted_confidence"] = float(probs[pred_idx])

            sorted_probs = np.sort(probs)[::-1]
            s["margin"] = float(sorted_probs[0] - sorted_probs[1])
            s["normalized_margin"] = float(s["margin"] / (sorted_probs[0] + 1e-8))

    def find_label_errors(self):
        errors = []
        for s in self.samples:
            if (
                s["predicted_label_idx"] != s["given_label_idx"]
                and s["predicted_confidence"] >= self.thresholds[s["predicted_label_idx"]]
            ):
                s["is_error"] = True
                errors.append(s.copy())
                print(
                    f"ERROR: {s['given_label']} → {s['predicted_label']} ({s['predicted_confidence']:.3f})"
                )
            else:
                s["is_error"] = False
        return errors

    def visualize_all(self, images_path, output_dir):
        if not self.samples:
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

        print("\n=== Voxel Visualizations ===")
        self.voxel_viz.visualize_voxel_space(
            self.samples, self.categories, os.path.join(voxel_dir, "space.html")
        )
        self.voxel_viz.visualize_category_voxels(
            self.samples, self.categories, os.path.join(voxel_dir, "categories.html")
        )
        self.voxel_viz.visualize_error_voxels(
            self.samples, os.path.join(voxel_dir, "errors.html")
        )
        self.voxel_viz.visualize_density_heatmap(
            self.samples, os.path.join(voxel_dir, "density.html")
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
        axes[0].set_title("Confident Joint")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Given")

        Q = self.estimate_latent_joint()
        sns.heatmap(
            Q,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[1],
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title("Normalized")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Given")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confident_joint.png"), dpi=300)
        plt.close()

    def _visualize_noise_matrices(self, output_dir):
        noise = self.estimate_noise_matrix()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(
            noise,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[0],
            vmin=0,
            vmax=1,
        )
        axes[0].set_title("Noise Matrix")

        sns.heatmap(
            noise.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.categories,
            yticklabels=self.categories,
            ax=axes[1],
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Inverse")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "noise_matrices.png"), dpi=300)
        plt.close()

    def _visualize_probability_distributions(self, output_dir):
        if not self.samples:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self_conf = [s.get("self_confidence", 0.5) for s in self.samples]

        axes[0, 0].hist(self_conf, bins=30, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Self-Confidence")
        axes[0, 0].axvline(
            np.mean(self_conf),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(self_conf):.3f}",
        )
        axes[0, 0].legend()

        margins = [s.get("margin", 0) for s in self.samples]
        axes[0, 1].hist(margins, bins=30, color="lightgreen", edgecolor="black")
        axes[0, 1].set_title("Margin")
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
            axes[1, 0].set_title("Correct vs Errors")
            axes[1, 0].legend()
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                f"{'No errors' if not errors else 'All errors'}",
                ha="center",
                va="center",
            )

        axes[1, 1].bar(range(len(self.categories)), list(self.thresholds), color="coral")
        axes[1, 1].set_xticks(range(len(self.categories)))
        axes[1, 1].set_xticklabels(self.categories, rotation=45, ha="right")
        axes[1, 1].set_title("Thresholds")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "probability_distributions.png"), dpi=300)
        plt.close()

    def _visualize_embeddings(self, output_dir):
        if len(self.samples) < 10:
            return

        X = self.pred_probs_matrix.copy()
        if np.isnan(X).any():
            imputer = SimpleImputer(
                strategy="constant", fill_value=1.0 / len(self.categories)
            )
            X = imputer.fit_transform(X)

        if np.isnan(X).any() or np.allclose(X, X[0]):
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
                axes[0, 0].set_title("t-SNE: Labels")
                plt.colorbar(scatter, ax=axes[0, 0])
                colors = ["green" if not e else "red" for e in is_error]
                axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=50)
                axes[0, 1].set_title("t-SNE: Errors")
            except:
                axes[0, 0].text(0.5, 0.5, "t-SNE failed", ha="center")
                axes[0, 1].text(0.5, 0.5, "t-SNE failed", ha="center")
        else:
            axes[0, 0].text(0.5, 0.5, f"Need 50+ samples (have {len(X)})", ha="center")
            axes[0, 1].text(0.5, 0.5, f"Need 50+ samples (have {len(X)})", ha="center")

        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            scatter = axes[1, 0].scatter(
                X_pca[:, 0], X_pca[:, 1], c=y_given, cmap="tab10", alpha=0.6, s=50
            )
            axes[1, 0].set_title(f"PCA: {sum(pca.explained_variance_ratio_):.2%} var")
            plt.colorbar(scatter, ax=axes[1, 0])
            colors = ["green" if not e else "red" for e in is_error]
            axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
            axes[1, 1].set_title("PCA: Errors")
        except:
            axes[1, 0].text(0.5, 0.5, "PCA failed", ha="center")
            axes[1, 1].text(0.5, 0.5, "PCA failed", ha="center")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "embeddings.png"), dpi=300)
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
        axes[0, 0].set_title("Label Quality")
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
        axes[0, 1].set_title("Margin")

        category_error_rates = df.groupby("given_label")["is_error"].mean()
        axes[1, 0].bar(
            range(len(category_error_rates)), category_error_rates.values, color="salmon"
        )
        axes[1, 0].set_xticks(range(len(category_error_rates)))
        axes[1, 0].set_xticklabels(category_error_rates.index, rotation=45, ha="right")
        axes[1, 0].set_title("Error Rate by Category")

        category_counts = df["given_label"].value_counts()
        axes[1, 1].bar(
            range(len(category_counts)), category_counts.values, color="steelblue"
        )
        axes[1, 1].set_xticks(range(len(category_counts)))
        axes[1, 1].set_xticklabels(category_counts.index, rotation=45, ha="right")
        axes[1, 1].set_title("Sample Distribution")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "label_quality_scores.png"), dpi=300)
        plt.close()

    def _visualize_error_examples(self, images_path, output_dir):
        errors = [s for s in self.samples if s.get("is_error", False)]
        if not errors:
            return

        errors_sorted = sorted(errors, key=lambda x: x["margin"])[:12]
        n_cols, n_rows = 4, min(3, (len(errors_sorted) + 3) // 4)
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
            title = f"Given: {error['given_label']}\nPred: {error['predicted_label']}\nConf: {error['predicted_confidence']:.2f}"
            axes[idx].set_title(title, fontsize=9)

        for idx in range(len(errors_sorted), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_examples.png"), dpi=300)
        plt.close()

    def save_results(self, images_path, output_dir="data/processed/confident_learning"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        self.compute_label_quality_scores()
        errors = self.find_label_errors()

        pd.DataFrame(self.samples).to_csv(
            os.path.join(run_dir, "all_samples.csv"), index=False
        )
        if errors:
            pd.DataFrame(errors).to_csv(os.path.join(run_dir, "errors.csv"), index=False)

        np.save(os.path.join(run_dir, "confident_joint.npy"), self.confident_joint)
        np.save(os.path.join(run_dir, "noise_matrix.npy"), self.estimate_noise_matrix())

        stats = {
            "total": len(self.samples),
            "errors": len(errors),
            "error_rate": len(errors) / len(self.samples) if self.samples else 0,
            "thresholds": {
                cat: float(t) for cat, t in zip(self.categories, self.thresholds)
            },
        }

        with open(os.path.join(run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        self.visualize_all(images_path, run_dir)

        print(f"\nResults: {run_dir}")
        print(f"Errors: {len(errors)}/{len(self.samples)} ({stats['error_rate']:.2%})")
        return errors


if __name__ == "__main__":
    model_path = r"C:\Users\nkz3kor\Documents\traffic_vlm\qwen-model"
    cl = BDD100KConfidentLearning(model_path)

    images_path = r"data\raw\mini-x\images\test"
    labels_path = r"data\raw\mini-x\labels\test"

    cl.process_dataset(images_path, labels_path, max_samples=50)
    cl.compute_confident_joint()
    errors = cl.save_results(images_path)
