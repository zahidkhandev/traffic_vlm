import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import NDArray
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Run: pip install umap-learn")


class QwenVLMValidator:
    def __init__(self, model_path: str):
        print(f"Loading Qwen3-VL model from {model_path}...")

        if torch.cuda.is_available():
            torch_dtype = torch.float16
            device_map = "cuda"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"

        print(f"Using device: {device_map} with dtype: {torch_dtype}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        self.object_classes = [
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

        print(f"Model loaded on {self.model.device}")

    def classify_crop_with_embedding(
        self, crop: Image.Image
    ) -> Tuple[Dict[str, float], NDArray[np.float32]]:
        w, h = crop.size
        if w > 384 or h > 384:
            resample_filter = Image.Resampling.LANCZOS
            crop = crop.resize((min(w, 384), min(h, 384)), resample_filter)

        class_list = ", ".join(self.object_classes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {
                        "type": "text",
                        "text": f"What object is this? Choose exactly one: {class_list}",
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_outputs = process_vision_info(messages)
        image_inputs = vision_outputs[0]
        video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            model_outputs = self.model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                output_hidden_states=True,
            )

            embedding = model_outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

            outputs = self.model.generate(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
            )

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        response_text = response.split("assistant")[-1].strip().lower()

        probs = {cls: 0.01 for cls in self.object_classes}

        for cls in self.object_classes:
            cls_variants = [cls, cls.replace(" ", ""), cls.replace(" ", "-")]
            for variant in cls_variants:
                if variant in response_text:
                    probs[cls] = 0.9
                    break

        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        return probs, embedding


class BDD100KConfidentLearning:
    def __init__(self, model_path: str, checkpoint_dir: str = "data/checkpoints"):
        self.vlm = QwenVLMValidator(model_path)
        self.object_classes = self.vlm.object_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.object_classes)}
        self.samples: List[Dict[str, Any]] = []
        self.embeddings: List[NDArray[np.float32]] = []
        self.crops_cache: Dict[int, Image.Image] = {}
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def load_checkpoint(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            self.samples = checkpoint["samples"]
            self.embeddings = checkpoint["embeddings"]
            processed_images = checkpoint["processed_images"]
            print(
                f"Loaded checkpoint: {len(self.samples)} samples from {len(processed_images)} images"
            )
            return processed_images
        return set()

    def save_checkpoint(self, processed_images: set):
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pkl")
        checkpoint = {
            "samples": self.samples,
            "embeddings": self.embeddings,
            "processed_images": processed_images,
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

    def process_dataset(
        self,
        images_path: str,
        labels_path: str,
        max_samples: Optional[int] = None,
        checkpoint_every: int = 10,
    ):
        processed_images = self.load_checkpoint()

        label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".json")])

        if max_samples is not None:
            label_files = label_files[:max_samples]

        print(
            f"\nProcessing {len(label_files)} images (skip {len(processed_images)} already processed)"
        )

        total_objects = len(self.samples)
        objects_since_checkpoint = 0
        pbar_images = tqdm(label_files, desc="Images", position=0)

        for label_file in pbar_images:
            file_id = os.path.splitext(label_file)[0]

            if file_id in processed_images:
                continue

            img_path = os.path.join(images_path, f"{file_id}.jpg")
            label_path = os.path.join(labels_path, label_file)

            if not os.path.exists(img_path):
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            full_img = Image.open(img_path).convert("RGB")
            img_width, img_height = full_img.size

            frames = label_data.get("frames", [])

            objects_to_process = []
            for frame in frames:
                objects = frame.get("objects", [])
                for obj in objects:
                    category = obj.get("category")

                    if "/" in category or category not in self.object_classes:
                        continue

                    box = obj.get("box2d")
                    if not box:
                        continue

                    x1 = max(0, int(box["x1"]))
                    y1 = max(0, int(box["y1"]))
                    x2 = min(img_width, int(box["x2"]))
                    y2 = min(img_height, int(box["y2"]))

                    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    objects_to_process.append(
                        {"obj": obj, "category": category, "box": [x1, y1, x2, y2]}
                    )

            pbar_objects = tqdm(
                objects_to_process,
                desc=f"  {file_id}",
                position=1,
                leave=False,
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            for obj_data in pbar_objects:
                obj = obj_data["obj"]
                category = obj_data["category"]
                x1, y1, x2, y2 = obj_data["box"]

                crop = full_img.crop((x1, y1, x2, y2))

                if total_objects == 0:
                    pbar_objects.write(
                        f"\n[First object: {category} at [{x1},{y1},{x2},{y2}] size={crop.size}]"
                    )
                    start = time.time()

                pred_probs, embedding = self.vlm.classify_crop_with_embedding(crop)

                if total_objects == 0:
                    elapsed = time.time() - start
                    predicted_class = max(pred_probs.items(), key=lambda item: item[1])[0]
                    pbar_objects.write(
                        f"[First inference: {elapsed:.1f}s | Pred: {predicted_class} | Given: {category}]\n"
                    )

                self.crops_cache[total_objects] = crop.copy()

                obj_attrs = obj.get("attributes", {})
                self.samples.append(
                    {
                        "sample_idx": total_objects,
                        "image": f"{file_id}.jpg",
                        "obj_id": obj.get("id", -1),
                        "given_label": category,
                        "given_label_idx": self.class_to_idx[category],
                        "box": [x1, y1, x2, y2],
                        "pred_probs": pred_probs,
                        "occluded": obj_attrs.get("occluded", False),
                        "truncated": obj_attrs.get("truncated", False),
                    }
                )
                self.embeddings.append(embedding)

                total_objects += 1
                objects_since_checkpoint += 1

                pbar_objects.set_postfix(
                    {
                        "pred": max(pred_probs, key=lambda k: pred_probs[k])[:3],
                        "total": total_objects,
                    },
                    refresh=True,
                )

            pbar_objects.close()
            processed_images.add(file_id)

            if objects_since_checkpoint >= checkpoint_every:
                self.save_checkpoint(processed_images)
                objects_since_checkpoint = 0

        pbar_images.close()
        self.save_checkpoint(processed_images)
        print(
            f"\nCollected {len(self.samples)} samples with {len(self.embeddings)} embeddings"
        )

    def compute_confident_joint(self):
        if not self.samples:
            return np.zeros((len(self.object_classes), len(self.object_classes)))

        n = len(self.object_classes)
        confident_joint = np.zeros((n, n))

        pred_probs_matrix = np.array(
            [[s["pred_probs"][cls] for cls in self.object_classes] for s in self.samples]
        )

        thresholds = np.mean(pred_probs_matrix, axis=0)

        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cls] for cls in self.object_classes])

            for pred_idx in range(n):
                if probs[pred_idx] >= thresholds[pred_idx]:
                    confident_joint[given_idx, pred_idx] += 1

        self.confident_joint = confident_joint
        self.thresholds = thresholds
        self.pred_probs_matrix = pred_probs_matrix

        print("\nComputed C̃(ỹ, y*)")

        return confident_joint

    def estimate_latent_joint(self):
        return self.confident_joint / (self.confident_joint.sum() + 1e-8)

    def estimate_noise_matrix(self):
        Q_joint = self.estimate_latent_joint()
        py = Q_joint.sum(axis=0)
        return Q_joint / (py + 1e-8)

    def compute_label_quality_scores(self):
        for sample in tqdm(self.samples, desc="Computing quality scores"):
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cls] for cls in self.object_classes])

            pred_idx = int(np.argmax(probs))

            sample["self_confidence"] = float(probs[given_idx])
            sample["predicted_label_idx"] = pred_idx
            sample["predicted_label"] = self.object_classes[pred_idx]
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

        print(
            f"Found {len(errors)} errors ({100 * len(errors) / len(self.samples):.1f}%)"
        )
        return errors

    def save_error_crops(self, errors: List[Dict[str, Any]], run_dir: str):
        error_crops_dir = os.path.join(run_dir, "label_errors_to_verify")
        os.makedirs(error_crops_dir, exist_ok=True)

        for idx, error in enumerate(tqdm(errors, desc="Saving error crops")):
            sample_idx = error["sample_idx"]

            if sample_idx in self.crops_cache:
                crop = self.crops_cache[sample_idx]

                filename = (
                    f"error_{idx:03d}_"
                    f"given_{error['given_label']}_"
                    f"pred_{error['predicted_label']}_"
                    f"conf_{error['predicted_confidence']:.3f}.jpg"
                )

                crop_path = os.path.join(error_crops_dir, filename)
                crop.save(crop_path, quality=95)

        print(f"Saved {len(errors)} error crops to {error_crops_dir}")

    def visualize_errors_grid(
        self, errors: List[Dict[str, Any]], output_dir: str, max_display: int = 50
    ):
        if not errors:
            print("No errors to visualize")
            return

        errors_sorted = sorted(
            errors, key=lambda x: x["predicted_confidence"], reverse=True
        )[:max_display]

        n_cols = 5
        n_rows = (len(errors_sorted) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, error in enumerate(tqdm(errors_sorted, desc="Creating error grid")):
            if idx >= len(axes_flat):
                break

            sample_idx = error["sample_idx"]
            if sample_idx in self.crops_cache:
                crop = self.crops_cache[sample_idx]
                axes_flat[idx].imshow(crop)
                axes_flat[idx].axis("off")

                title = (
                    f"✗ ERROR #{idx + 1}\n"
                    f"Given: {error['given_label']}\n"
                    f"Predicted: {error['predicted_label']}\n"
                    f"Conf: {error['predicted_confidence']:.3f}"
                )
                axes_flat[idx].set_title(
                    title, fontsize=9, color="red", fontweight="bold"
                )

        for idx in range(len(errors_sorted), len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "error_grid.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()

        print("Saved error grid")

    def visualize_embedding_umap(self, output_dir: str):
        if not HAS_UMAP:
            print("Skipping UMAP (not installed)")
            return

        if len(self.embeddings) < 5:
            print(f"Skipping UMAP (need ≥5 samples, have {len(self.embeddings)})")
            return

        embeddings_array = np.array(self.embeddings)

        print(f"\nComputing UMAP for {len(embeddings_array)} samples...")
        n_neighbors = min(15, len(embeddings_array) - 1)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42
        )
        embedding_2d = np.array(reducer.fit_transform(embeddings_array))

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        labels_given = [s["given_label"] for s in self.samples]
        labels_predicted = [s["predicted_label"] for s in self.samples]
        is_error = np.array([s.get("is_error", False) for s in self.samples])

        n_classes = len(self.object_classes)
        cmap = plt.get_cmap(
            "tab10" if n_classes <= 10 else "tab20" if n_classes <= 20 else "hsv"
        )
        label_to_color = {
            cls: cmap(i % cmap.N) for i, cls in enumerate(self.object_classes)
        }

        for label in set(labels_given):
            mask = np.array([lbl == label for lbl in labels_given])
            axes[0].scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.7,
                s=50,
            )
        axes[0].set_title("UMAP: Given Labels", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=8, loc="best")
        axes[0].grid(True, alpha=0.3)

        for label in set(labels_predicted):
            mask = np.array([lbl == label for lbl in labels_predicted])
            axes[1].scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.7,
                s=50,
            )
        axes[1].set_title("UMAP: Predicted Labels", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=8, loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(
            embedding_2d[~is_error, 0],
            embedding_2d[~is_error, 1],
            c="green",
            label="Correct",
            alpha=0.5,
            s=50,
        )
        axes[2].scatter(
            embedding_2d[is_error, 0],
            embedding_2d[is_error, 1],
            c="red",
            label="Error",
            alpha=0.8,
            s=100,
            marker="X",
        )
        axes[2].set_title("UMAP: Label Quality", fontsize=14, fontweight="bold")
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "umap_embeddings.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("Saved UMAP visualization")

    def visualize_confident_joint(self, output_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            self.confident_joint,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[0],
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title("C̃(ỹ, y*)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Predicted y*")
        axes[0].set_ylabel("Given ỹ")

        Q_joint = self.estimate_latent_joint()
        sns.heatmap(
            Q_joint,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[1],
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title("Q̃(ỹ, y*)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Predicted y*")
        axes[1].set_ylabel("Given ỹ")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confident_joint.png"), dpi=300)
        plt.close()

    def visualize_noise_matrix(self, output_dir: str):
        noise_matrix = self.estimate_noise_matrix()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            noise_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[0],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "P(ỹ|y*)"},
        )
        axes[0].set_title("P(ỹ|y*)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("True y*")
        axes[0].set_ylabel("Noisy ỹ")

        sns.heatmap(
            noise_matrix.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "P(y*|ỹ)"},
        )
        axes[1].set_title("P(y*|ỹ)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Noisy ỹ")
        axes[1].set_ylabel("True y*")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "noise_matrix.png"), dpi=300)
        plt.close()

    def save_results(self, output_dir: str = "data/processed/confident_learning"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        self.compute_label_quality_scores()
        errors = self.find_label_errors()

        if errors:
            errors_df = pd.DataFrame(errors)
            errors_df.to_csv(os.path.join(run_dir, "label_errors.csv"), index=False)
            self.save_error_crops(errors, run_dir)

        pd.DataFrame(self.samples).to_csv(
            os.path.join(run_dir, "all_samples.csv"), index=False
        )

        np.save(os.path.join(run_dir, "confident_joint.npy"), self.confident_joint)
        np.save(os.path.join(run_dir, "noise_matrix.npy"), self.estimate_noise_matrix())
        np.save(os.path.join(run_dir, "embeddings.npy"), np.array(self.embeddings))

        stats = {
            "total_samples": len(self.samples),
            "label_errors": len(errors),
            "error_rate": len(errors) / len(self.samples) if self.samples else 0,
            "thresholds": {
                cls: float(t) for cls, t in zip(self.object_classes, self.thresholds)
            },
        }

        with open(os.path.join(run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        viz_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        print("\nGenerating visualizations...")
        self.visualize_confident_joint(viz_dir)
        self.visualize_noise_matrix(viz_dir)
        self.visualize_errors_grid(errors, viz_dir)
        self.visualize_embedding_umap(viz_dir)

        self.crops_cache.clear()

        print(f"\nResults: {run_dir}")
        print(
            f"Samples: {stats['total_samples']} | Errors: {stats['label_errors']} ({stats['error_rate']:.2%})"
        )

        return errors


if __name__ == "__main__":
    print("BDD100K Confident Learning - Error Verification Mode")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = "qwen-vl-4b"
    cl = BDD100KConfidentLearning(model_path)

    images_path = r"data\raw\mini\images\test"
    labels_path = r"data\raw\mini\labels\test"

    cl.process_dataset(images_path, labels_path, max_samples=None, checkpoint_every=10)
    cl.compute_confident_joint()
    errors = cl.save_results()

    print(f"\nDONE! {len(errors)} label errors found")
    print(
        "Check errors in: data/processed/confident_learning/run_*/label_errors_to_verify/"
    )
