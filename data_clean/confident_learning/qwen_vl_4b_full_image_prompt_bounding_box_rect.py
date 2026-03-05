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
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Run: pip install umap-learn")


CLASS_TOKEN_ALIASES = {
    "traffic light": "trafficlight",
    "traffic sign": "trafficsign",
}
CLASS_ALIAS_REVERSE = {v: k for k, v in CLASS_TOKEN_ALIASES.items()}


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

        self.object_classes_prompt = [
            CLASS_TOKEN_ALIASES.get(cls, cls) for cls in self.object_classes
        ]

        self.class_first_token_ids = []
        for alias in self.object_classes_prompt:
            token_id = self.processor.tokenizer(
                alias, add_special_tokens=False
            ).input_ids[0]
            self.class_first_token_ids.append(token_id)

        if len(self.class_first_token_ids) != len(set(self.class_first_token_ids)):
            pairs = list(zip(self.object_classes_prompt, self.class_first_token_ids))
            print("TOKEN ID MAP:", pairs)
            raise ValueError(
                "Token ID collision detected! Two classes map to the same first token. "
                "Change CLASS_TOKEN_ALIASES until all IDs are unique."
            )

        print("Token ID map (alias → id):")
        for alias, tid in zip(self.object_classes_prompt, self.class_first_token_ids):
            print(f"  {alias:20s} → {tid}")

        print(f"Model loaded on {self.model.device}")

    def _run_inference(self, messages: List[Dict]) -> Tuple[Dict[str, float], NDArray]:
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

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        with torch.no_grad():
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )
            embedding = model_outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

            gen_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        response = self.processor.batch_decode(
            gen_outputs.sequences,  # type: ignore
            skip_special_tokens=True,
        )[0]
        response_text = response.split("assistant")[-1].strip()

        first_token_logits = gen_outputs.scores[0][0]  # type: ignore

        response_lower = response_text.lower().strip()
        parsed_class = None

        for alias, original in CLASS_ALIAS_REVERSE.items():
            if alias in response_lower:
                parsed_class = original
                break

        if parsed_class is None:
            for cls in self.object_classes:
                if cls.lower() in response_lower:
                    parsed_class = cls
                    break

        class_logits_list = []
        for token_id in self.class_first_token_ids:
            if token_id < len(first_token_logits):
                class_logits_list.append(float(first_token_logits[token_id]))
            else:
                class_logits_list.append(0.0)

        class_logits_tensor = torch.tensor(
            class_logits_list, device=self.model.device, dtype=torch.float32
        )
        class_probs = (
            torch.nn.functional.softmax(class_logits_tensor, dim=0).cpu().numpy()
        )
        probs = {cls: float(prob) for cls, prob in zip(self.object_classes, class_probs)}

        if parsed_class is None:
            parsed_class = self.object_classes[int(np.argmax(class_probs))]

        predicted_class = parsed_class
        max_conf = float(probs[predicted_class])

        print(
            f"  [VLM] -> '{response_text}' (Parsed: {predicted_class} at {max_conf:.3f})"
        )

        return probs, embedding

    def classify_object(
        self, full_img: Image.Image, box: List[int], use_grounding: bool = True
    ) -> Tuple[Dict[str, float], NDArray]:
        x1, y1, x2, y2 = box

        if use_grounding:
            img_to_send = full_img.copy()
            draw = ImageDraw.Draw(img_to_send)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            class_names_str = ", ".join(self.object_classes_prompt)
            prompt = (
                f"What is the object inside the red box? "
                f"Choose from: {class_names_str}.\n\n"
                f"Reply with only a single word from the list above."
            )
        else:
            img_to_send = full_img.crop((x1, y1, x2, y2))

            class_options = "\n".join(
                [f"{i}: {alias}" for i, alias in enumerate(self.object_classes_prompt)]
            )
            prompt = (
                f"Identify the main object in this image from the following list:\n"
                f"{class_options}\n\n"
                f"Output EXACTLY ONE digit (0-9) corresponding to the correct class. "
                f"Do not write any other text or explanation."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_to_send,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        return self._run_inference(messages)


class BDD100KConfidentLearning:
    def __init__(
        self,
        model_path: str,
        checkpoint_dir: str = "data/checkpoints",
        debug: bool = False,
        debug_output_dir: str = "data/processed/confident_learning/run_with_bounding_box_red_rect",
        use_grounding: bool = True,
    ):
        self.vlm = QwenVLMValidator(model_path)
        self.object_classes = self.vlm.object_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.object_classes)}

        self.samples: List[Dict[str, Any]] = []
        self.embeddings: List[NDArray] = []
        self.crops_cache: Dict[int, Image.Image] = {}

        self.confident_joint: Optional[NDArray] = None
        self.thresholds: Optional[NDArray] = None
        self.pred_probs_matrix: Optional[NDArray] = None

        self.checkpoint_dir = checkpoint_dir
        self.debug = debug
        self.debug_output_dir = debug_output_dir
        self.run_dir = None
        self.use_grounding = use_grounding
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.debug_occupied_regions: Dict[
            str, List[Tuple[float, float, float, float]]
        ] = {}

        if self.debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(debug_output_dir, f"run_{timestamp}")
            self.debug_dir = os.path.join(self.run_dir, "debug_images")
            os.makedirs(self.debug_dir, exist_ok=True)
            mode = "grounding (full image + box)" if use_grounding else "crop"
            print(f"Debug mode ON - Inference mode: {mode}")
        else:
            self.debug_dir = None
            mode = "grounding (full image + box)" if use_grounding else "crop"
            print(f"Inference mode: {mode}")

    def _save_debug_image(
        self,
        full_img: Image.Image,
        box: List[int],
        given_label: str,
        pred_label: str,
        pred_conf: float,
        file_id: str,
        obj_count: int,
    ) -> None:
        if not self.debug or self.debug_dir is None:
            return

        x1, y1, x2, y2 = box
        label_text = (
            f"{obj_count}. Given: {given_label} | Pred: {pred_label} ({pred_conf:.2f})"
        )

        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        font = dummy_draw.getfont()
        t_bbox = dummy_draw.textbbox((0, 0), label_text, font=font)
        text_width = t_bbox[2] - t_bbox[0]
        text_height = t_bbox[3] - t_bbox[1]

        margin = 2
        version = 0

        while True:
            filename = f"{file_id}_all_objects{f'_v{version}' if version > 0 else ''}.png"
            filepath = os.path.join(self.debug_dir, filename)

            if filepath not in self.debug_occupied_regions:
                self.debug_occupied_regions[filepath] = []

            occupied = self.debug_occupied_regions[filepath]

            candidates = [
                (x1, y1 - text_height - margin * 2),
                (x1, y2 + margin * 2),
                (x1, y1 + margin * 2),
                (x1, y2 - text_height - margin * 2),
            ]

            placed = False
            for px, py in candidates:
                px = max(0, px)
                py = max(0, py)

                proposed_box = (
                    px - margin,
                    py - margin,
                    px + text_width + margin,
                    py + text_height + margin,
                )

                collision = False
                for ox1, oy1, ox2, oy2 in occupied:
                    if not (
                        proposed_box[2] < ox1
                        or proposed_box[0] > ox2
                        or proposed_box[3] < oy1
                        or proposed_box[1] > oy2
                    ):
                        collision = True
                        break

                if not collision:
                    self.debug_occupied_regions[filepath].append(proposed_box)

                    if os.path.exists(filepath):
                        img_copy = Image.open(filepath)
                    else:
                        img_copy = full_img.copy()

                    draw = ImageDraw.Draw(img_copy)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                    draw.rectangle(proposed_box, fill="white", outline="red")
                    draw.text((px, py), label_text, fill="red")

                    img_copy.save(filepath)
                    placed = True
                    break

            if placed:
                break
            else:
                version += 1

    def save_checkpoint(self, processed_images: set) -> None:
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pkl")
        checkpoint_data = {
            "samples": self.samples,
            "embeddings": self.embeddings,
            "processed_images": processed_images,
            "confident_joint": self.confident_joint,
            "thresholds": self.thresholds,
            "pred_probs_matrix": self.pred_probs_matrix,
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(
            f"Checkpoint saved: {len(self.samples)} samples, {len(processed_images)} images"
        )

    def load_checkpoint(self) -> set:
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pkl")
        if not os.path.exists(checkpoint_file):
            return set()

        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        self.samples = checkpoint.get("samples", [])
        self.embeddings = checkpoint.get("embeddings", [])
        self.confident_joint = checkpoint.get("confident_joint")
        self.thresholds = checkpoint.get("thresholds")
        self.pred_probs_matrix = checkpoint.get("pred_probs_matrix")

        processed_images = checkpoint.get("processed_images", set())

        if len(self.samples) != len(self.embeddings):
            print(
                f"WARNING: Sample/embedding mismatch! "
                f"Samples: {len(self.samples)}, Embeddings: {len(self.embeddings)}"
            )
            min_len = min(len(self.samples), len(self.embeddings))
            self.samples = self.samples[:min_len]
            self.embeddings = self.embeddings[:min_len]

        cj_status = "ready" if self.confident_joint is not None else "not computed"
        print(
            f"Checkpoint loaded: {len(self.samples)} samples, "
            f"{len(processed_images)} images, CJ={cj_status}"
        )
        return processed_images

    def process_dataset(
        self,
        images_path: str,
        labels_path: str,
        max_samples: Optional[int] = None,
        checkpoint_every: int = 10,
    ) -> None:
        processed_images = self.load_checkpoint()
        label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".json")])

        if max_samples is not None:
            label_files = label_files[:max_samples]

        print(
            f"Processing {len(label_files)} images "
            f"({len(processed_images)} already done, {len(self.samples)} objects accumulated)"
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

            objects_to_process = []
            for frame in label_data.get("frames", []):
                for obj in frame.get("objects", []):
                    category = obj.get("category")
                    if category not in self.object_classes:
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

                if total_objects == 0:
                    pbar_objects.write(
                        f"First object: {category} at [{x1},{y1},{x2},{y2}]"
                    )
                    start = time.time()

                pred_probs, embedding = self.vlm.classify_object(
                    full_img, [x1, y1, x2, y2], use_grounding=self.use_grounding
                )

                if total_objects == 0:
                    elapsed = time.time() - start
                    predicted_class = max(pred_probs, key=lambda k: pred_probs[k])
                    pbar_objects.write(
                        f"First inference: {elapsed:.1f}s | Pred: {predicted_class} | Given: {category}"
                    )

                crop = full_img.crop((x1, y1, x2, y2))
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

                if self.debug:
                    pred_label = max(pred_probs, key=lambda k: pred_probs[k])
                    pred_conf = pred_probs[pred_label]
                    obj_num = len(
                        [s for s in self.samples if s["image"] == f"{file_id}.jpg"]
                    )
                    self._save_debug_image(
                        full_img,
                        [x1, y1, x2, y2],
                        category,
                        pred_label,
                        pred_conf,
                        file_id,
                        obj_num,
                    )

                total_objects += 1
                objects_since_checkpoint += 1

                pbar_objects.set_postfix(
                    {
                        "pred": max(pred_probs, key=lambda k: pred_probs[k])[:6],
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
            f"Collection complete: {len(self.samples)} samples, "
            f"{len(self.embeddings)} embeddings"
        )

    def compute_confident_joint(self) -> NDArray:
        if not self.samples:
            return np.zeros((len(self.object_classes), len(self.object_classes)))

        n = len(self.object_classes)
        pred_probs_matrix = np.array(
            [[s["pred_probs"][cls] for cls in self.object_classes] for s in self.samples]
        )

        thresholds = np.percentile(pred_probs_matrix, 80, axis=0)
        print(
            "Thresholds (80th percentile):",
            {cls: f"{t:.3f}" for cls, t in zip(self.object_classes, thresholds)},
        )

        confident_joint = np.zeros((n, n))
        for sample in self.samples:
            given_idx = sample["given_label_idx"]
            probs = np.array([sample["pred_probs"][cls] for cls in self.object_classes])
            for pred_idx in range(n):
                if probs[pred_idx] >= thresholds[pred_idx]:
                    confident_joint[given_idx, pred_idx] += 1

        self.confident_joint = confident_joint
        self.thresholds = thresholds
        self.pred_probs_matrix = pred_probs_matrix

        print(f"Confident joint computed from {len(self.samples)} samples")
        return confident_joint

    def estimate_latent_joint(self) -> NDArray:
        if self.confident_joint is None:
            raise ValueError("Call compute_confident_joint() first")
        return self.confident_joint / (self.confident_joint.sum() + 1e-8)

    def estimate_noise_matrix(self) -> NDArray:
        Q_joint = self.estimate_latent_joint()
        py = Q_joint.sum(axis=0)
        return Q_joint / (py + 1e-8)

    def compute_label_quality_scores(self) -> None:
        if self.thresholds is None:
            raise ValueError("Call compute_confident_joint() first")

        for sample in tqdm(self.samples, desc="Computing quality scores"):
            probs = np.array([sample["pred_probs"][cls] for cls in self.object_classes])
            pred_idx = int(np.argmax(probs))
            given_idx = sample["given_label_idx"]

            sorted_probs = np.sort(probs)[::-1]

            sample["self_confidence"] = float(probs[given_idx])
            sample["predicted_label_idx"] = pred_idx
            sample["predicted_label"] = self.object_classes[pred_idx]
            sample["predicted_confidence"] = float(probs[pred_idx])
            sample["margin"] = float(sorted_probs[0] - sorted_probs[1])
            sample["normalized_margin"] = float(sorted_probs[0] - sorted_probs[1]) / (
                sorted_probs[0] + 1e-8
            )

    def find_label_errors(self) -> List[Dict[str, Any]]:
        if self.thresholds is None:
            raise ValueError("Call compute_confident_joint() first")

        errors = []
        for sample in self.samples:
            pred_idx = sample["predicted_label_idx"]
            given_idx = sample["given_label_idx"]
            pred_conf = sample["predicted_confidence"]
            margin = sample["margin"]

            is_error = (
                pred_idx != given_idx
                and pred_conf >= self.thresholds[pred_idx]
                and margin > 0.20
            )

            sample["is_error"] = is_error
            if is_error:
                errors.append(sample.copy())

        error_rate = 100 * len(errors) / len(self.samples) if self.samples else 0
        print(f"Found {len(errors)} errors ({error_rate:.1f}%) with 3-gate filter")
        return errors

    def save_error_crops(self, errors: List[Dict[str, Any]], run_dir: str) -> None:
        error_crops_dir = os.path.join(run_dir, "label_errors_to_verify")
        os.makedirs(error_crops_dir, exist_ok=True)

        for idx, error in enumerate(tqdm(errors, desc="Saving error crops")):
            sample_idx = error["sample_idx"]
            if sample_idx not in self.crops_cache:
                continue
            crop = self.crops_cache[sample_idx]
            filename = (
                f"error_{idx:03d}_"
                f"given_{error['given_label'].replace(' ', '_')}_"
                f"pred_{error['predicted_label'].replace(' ', '_')}_"
                f"conf_{error['predicted_confidence']:.3f}.png"
            )
            crop.save(os.path.join(error_crops_dir, filename))

        print(f"Saved {len(errors)} error crops to {error_crops_dir}")

    def visualize_errors_grid(
        self, errors: List[Dict[str, Any]], output_dir: str, max_display: int = 50
    ) -> None:
        if not errors:
            print("No errors to visualize")
            return

        errors_sorted = sorted(
            errors, key=lambda x: x["predicted_confidence"], reverse=True
        )[:max_display]

        n_cols = 5
        n_rows = (len(errors_sorted) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows), dpi=150)

        if n_rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, error in enumerate(tqdm(errors_sorted, desc="Creating error grid")):
            if idx >= len(axes_flat):
                break
            sample_idx = error["sample_idx"]
            if sample_idx not in self.crops_cache:
                continue

            crop = self.crops_cache[sample_idx]
            ax = axes_flat[idx]
            ax.imshow(crop)
            ax.axis("off")
            ax.set_title(
                f"ERROR {idx + 1}\nGiven: {error['given_label']}\n"
                f"Pred: {error['predicted_label']}\nConf: {error['predicted_confidence']:.3f}",
                fontsize=8,
                color="black",
                fontweight="bold",
                pad=4,
            )

        for idx in range(len(errors_sorted), len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.suptitle(
            f"Label Errors - {len(errors_sorted)} shown (sorted by confidence)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, "error_grid.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved error grid to {out_path}")

    def visualize_embedding_umap(self, output_dir: str) -> None:
        if not HAS_UMAP:
            print("Skipping UMAP - install umap-learn")
            return
        if len(self.embeddings) < 5:
            print("Skipping UMAP - need 5+ samples")
            return

        embeddings_array = np.array(self.embeddings)
        print(f"Computing UMAP for {len(embeddings_array)} samples...")
        n_neighbors = min(15, len(embeddings_array) - 1)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42
        )
        embedding_2d = np.array(reducer.fit_transform(embeddings_array))

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        labels_given = [s["given_label"] for s in self.samples]
        labels_predicted = [s["predicted_label"] for s in self.samples]
        is_error = np.array([s.get("is_error", False) for s in self.samples])

        cmap = plt.get_cmap("tab10")
        label_to_color = {cls: cmap(i % 10) for i, cls in enumerate(self.object_classes)}

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
        axes[0].legend(fontsize=8)
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
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(
            embedding_2d[~is_error, 0],
            embedding_2d[~is_error, 1],
            c="black",
            label="Correct",
            alpha=0.5,
            s=50,
        )
        axes[2].scatter(
            embedding_2d[is_error, 0],
            embedding_2d[is_error, 1],
            c="gray",
            label="Error",
            alpha=0.9,
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

    def visualize_confident_joint(self, output_dir: str) -> None:
        if self.confident_joint is None:
            print("Confident joint not computed")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        sns.heatmap(
            self.confident_joint,
            annot=True,
            fmt=".0f",
            cmap="Greys",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[0],
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title("Confident Joint (Raw Counts)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Predicted Class")
        axes[0].set_ylabel("Given (Noisy) Label")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        Q_joint = self.estimate_latent_joint()
        sns.heatmap(
            Q_joint,
            annot=True,
            fmt=".3f",
            cmap="Greys",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[1],
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title("Latent Joint (Normalized)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Predicted Class")
        axes[1].set_ylabel("Given (Noisy) Label")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "confident_joint.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("Saved confident joint visualization")

    def visualize_noise_matrix(self, output_dir: str) -> None:
        noise_matrix = self.estimate_noise_matrix()
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        sns.heatmap(
            noise_matrix,
            annot=True,
            fmt=".3f",
            cmap="Greys",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[0],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
        )
        axes[0].set_title(
            "Noise Matrix P(given=s | true=y)", fontsize=13, fontweight="bold"
        )
        axes[0].set_xlabel("True Label")
        axes[0].set_ylabel("Noisy Label")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        sns.heatmap(
            noise_matrix.T,
            annot=True,
            fmt=".3f",
            cmap="Greys",
            xticklabels=self.object_classes,
            yticklabels=self.object_classes,
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
        )
        axes[1].set_title(
            "Noise Matrix Transposed P(true=y | given=s)", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Noisy Label")
        axes[1].set_ylabel("True Label")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "noise_matrix.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("Saved noise matrix visualization")

    def save_results(
        self, output_dir: str = "data/processed/confident_learning"
    ) -> List[Dict[str, Any]]:
        if self.debug and self.run_dir is not None:
            run_dir = self.run_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(output_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

        self.compute_label_quality_scores()
        errors = self.find_label_errors()

        if errors:
            pd.DataFrame(errors).to_csv(
                os.path.join(run_dir, "label_errors.csv"), index=False
            )
            self.save_error_crops(errors, run_dir)

        pd.DataFrame(self.samples).to_csv(
            os.path.join(run_dir, "all_samples.csv"), index=False
        )

        if self.confident_joint is not None:
            np.save(os.path.join(run_dir, "confident_joint.npy"), self.confident_joint)
        np.save(os.path.join(run_dir, "noise_matrix.npy"), self.estimate_noise_matrix())
        np.save(os.path.join(run_dir, "embeddings.npy"), np.array(self.embeddings))

        stats = {
            "total_samples": len(self.samples),
            "label_errors": len(errors),
            "error_rate": len(errors) / len(self.samples) if self.samples else 0,
            "inference_mode": "grounding_tokens_full_image"
            if self.use_grounding
            else "crop_only",
            "thresholds": {
                cls: float(t) for cls, t in zip(self.object_classes, self.thresholds)
            }
            if self.thresholds is not None
            else {},
        }
        with open(os.path.join(run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        viz_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        print("Generating visualizations...")
        self.visualize_confident_joint(viz_dir)
        self.visualize_noise_matrix(viz_dir)
        self.visualize_errors_grid(errors, viz_dir)
        self.visualize_embedding_umap(viz_dir)

        self.crops_cache.clear()

        print(f"Results saved to {run_dir}")
        print(
            f"Samples: {stats['total_samples']} | Errors: {stats['label_errors']} ({stats['error_rate']:.2%})"
        )
        return errors


if __name__ == "__main__":
    print("=" * 80)
    print("BDD100K Confident Learning")
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = "qwen-vl-4b"

    cl = BDD100KConfidentLearning(
        model_path=model_path,
        checkpoint_dir="data/checkpoints",
        debug=True,
        use_grounding=True,
    )

    images_path = r"data\raw\mini\images\test"
    labels_path = r"data\raw\mini\labels\test"

    print("\nStarting dataset processing...")
    cl.process_dataset(images_path, labels_path, max_samples=None, checkpoint_every=10)

    print("\nComputing confident joint...")
    cl.compute_confident_joint()

    print("\nSaving results and generating visualizations...")
    errors = cl.save_results()

    print("\n" + "=" * 80)
    print(f"COMPLETE - Found {len(errors)} label errors")
    print("=" * 80)
