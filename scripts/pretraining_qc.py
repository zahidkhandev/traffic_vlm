import json
import os
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config.dataset_config import DatasetConfig
from data.vlm_qc import VLMJudge


class PreTrainingQC:
    def __init__(self, sample_rate=0.10):
        self.cfg = DatasetConfig()
        local_model_path = r"C:\Users\nkz3kor\Documents\traffic_vlm\llava-model"
        self.vlm_judge = VLMJudge(model_name=local_model_path)
        self.sample_rate = sample_rate

        self.audit_dir = os.path.join(self.cfg.output_dir, "qc_audit")
        if os.path.exists(self.audit_dir):
            shutil.rmtree(self.audit_dir)
        os.makedirs(self.audit_dir, exist_ok=True)

    def validate_split(self, split_name):
        print(f"\n{'=' * 60}")
        print(f"PRE-TRAINING QC: {split_name}")
        print(f"{'=' * 60}")

        cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")

        with open(cmd_path, "r") as f:
            commands = json.load(f)

        n_samples = int(len(commands) * self.sample_rate)
        sample_indices = random.sample(
            range(len(commands)), min(n_samples, len(commands))
        )

        print(
            f"Validating {len(sample_indices)} samples ({self.sample_rate * 100:.0f}%)..."
        )
        print(f"Audit images will be saved to: {self.audit_dir}")

        corrections = {}
        stats = {"correct": 0, "wrong": 0}

        with h5py.File(h5_path, "r") as hf:
            images_ds = hf["images"]
            metadata_ds = hf["metadata"]

            if not isinstance(images_ds, h5py.Dataset):
                raise TypeError("Images dataset not loaded")

            for idx in tqdm(sample_indices, desc=f"VLM Validation ({split_name})"):
                cmd = commands[idx]
                img_idx = cmd["image_idx"]

                img_array = np.array(images_ds[img_idx])
                image = Image.fromarray(img_array.astype("uint8"))

                meta_bytes = metadata_ds[img_idx]  # type: ignore
                if isinstance(meta_bytes, bytes):
                    meta_str = meta_bytes.decode("utf-8")
                elif isinstance(meta_bytes, np.bytes_):
                    meta_str = meta_bytes.tobytes().decode("utf-8")
                else:
                    meta_str = str(meta_bytes)

                meta = json.loads(meta_str)
                objects = meta.get("frames", [{}])[0].get("objects", [])

                is_valid, confidence, reason = self.vlm_judge.validate_label(
                    image, objects, cmd["a"]
                )

                if not is_valid:
                    suggested_fix = self._suggest_fix(reason, objects)

                    if suggested_fix == cmd["a"]:
                        stats["correct"] += 1
                        continue

                    stats["wrong"] += 1
                    corrections[idx] = {
                        "old_label": cmd["a"],
                        "new_label": suggested_fix,
                        "reason": reason,
                        "confidence": confidence,
                    }

                    self._save_audit_image(
                        image, split_name, idx, cmd["a"], suggested_fix, reason
                    )

                    tqdm.write(f"\n[ERROR] ID {idx}")
                    tqdm.write(f"  Old: {cmd['a']} -> New: {suggested_fix}")
                    tqdm.write(f"  VLM Reason: {reason}")
                    tqdm.write("-" * 40)

                else:
                    stats["correct"] += 1

        print(f"\n{'=' * 60}")
        print(f"VALIDATION RESULTS ({split_name})")
        print(f"Correct: {stats['correct']} | Wrong: {stats['wrong']}")
        print(f"Audit images saved to: {self.audit_dir}")
        print(f"{'=' * 60}\n")

        if corrections:
            corrected_commands = self._apply_corrections_with_heuristics(
                commands, corrections, sample_indices
            )
            output_path = os.path.join(
                self.cfg.output_dir, f"{split_name}_commands_qc.json"
            )
            with open(output_path, "w") as f:
                json.dump(corrected_commands, f, indent=2)
            print(f"Saved corrected labels to {output_path}")
            return corrections
        else:
            print("No corrections needed.")
            return {}

    def _save_audit_image(self, image, split, idx, old, new, reason):
        audit_img = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(audit_img)

        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

        text = f"ID: {idx}\nOLD: {old}\nNEW: {new}\n\nReason:\n{reason}"

        box_height = 300
        draw.rectangle([(0, 0), (1024, box_height)], fill=(0, 0, 0, 200))

        draw.text((20, 20), text, fill="red", font=font)

        filename = f"{split}_{idx}_error.jpg"
        audit_img.save(os.path.join(self.audit_dir, filename))

    def _suggest_fix(self, reason, objects):
        reason_lower = reason.lower()
        if "red light" in reason_lower or "traffic light" in reason_lower:
            return "stop_red_light"
        elif "pedestrian" in reason_lower or "person" in reason_lower:
            return "stop_pedestrian"
        elif "vehicle" in reason_lower or "car" in reason_lower:
            return "stop_vehicle"
        elif "obstacle" in reason_lower or "barrier" in reason_lower:
            return "stop_obstacle"
        else:
            return "safe"

    def _apply_corrections_with_heuristics(self, commands, corrections, sample_indices):
        corrected = commands.copy()
        for idx, correction in corrections.items():
            corrected[idx]["a"] = correction["new_label"]
            corrected[idx]["qc_corrected"] = True
            corrected[idx]["qc_reason"] = correction["reason"]

        print(f"Applied {len(corrections)} direct corrections")
        return corrected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=float, default=0.10)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    qc = PreTrainingQC(sample_rate=args.sample_rate)
    for split in args.splits:
        qc.validate_split(split)
