"""
Pre-Training QC: Validate CommandGenerator labels BEFORE first training
Uses VLM to check a sample of labels and fix obvious errors
"""

import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from config.dataset_config import DatasetConfig
from data.vlm_qc import VLMJudge


class PreTrainingQC:
    """Validates CommandGenerator labels before training starts"""

    def __init__(self, sample_rate=0.10):
        """
        Args:
            sample_rate: % of dataset to validate (0.10 = 10%)
        """
        self.cfg = DatasetConfig()
        self.vlm_judge = VLMJudge()
        self.sample_rate = sample_rate

    def validate_split(self, split_name):
        """
        Validates labels for a split and creates corrected version

        Returns:
            corrections: Dict of corrected labels
        """
        print(f"\n{'=' * 60}")
        print(f"PRE-TRAINING QC: {split_name}")
        print(f"{'=' * 60}")

        cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")

        # Load commands
        with open(cmd_path, "r") as f:
            commands = json.load(f)

        print(f"Total commands: {len(commands)}")

        # Sample for validation (e.g., 10%)
        n_samples = int(len(commands) * self.sample_rate)
        sample_indices = random.sample(
            range(len(commands)), min(n_samples, len(commands))
        )

        print(
            f"Validating {len(sample_indices)} samples ({self.sample_rate * 100:.0f}%)..."
        )

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

                # Load image
                img_array = np.array(images_ds[img_idx])
                image = Image.fromarray(img_array.astype("uint8"))

                # Load metadata
                meta_bytes = metadata_ds[img_idx]  # type: ignore
                if isinstance(meta_bytes, bytes):
                    meta_str = meta_bytes.decode("utf-8")
                elif isinstance(meta_bytes, np.bytes_):
                    meta_str = meta_bytes.tobytes().decode("utf-8")
                else:
                    meta_str = str(meta_bytes)

                meta = json.loads(meta_str)
                objects = meta.get("frames", [{}])[0].get("objects", [])

                # VLM validation
                is_valid, confidence, reason = self.vlm_judge.validate_label(
                    image, objects, cmd["a"]
                )

                if not is_valid:
                    stats["wrong"] += 1
                    suggested_fix = self._suggest_fix(reason, objects)

                    corrections[idx] = {
                        "old_label": cmd["a"],
                        "new_label": suggested_fix,
                        "reason": reason,
                        "confidence": confidence,
                    }

                    print(f"\n[{idx}] Label Error Found:")
                    print(f"  Original: {cmd['a']}")
                    print(f"  Suggested: {suggested_fix}")
                    print(f"  Reason: {reason}")
                else:
                    stats["correct"] += 1

        # Print statistics
        print(f"\n{'=' * 60}")
        print(f"VALIDATION RESULTS ({split_name})")
        print(f"{'=' * 60}")
        print(
            f"Correct: {stats['correct']} ({stats['correct'] / len(sample_indices) * 100:.1f}%)"
        )
        print(
            f"Errors Found: {stats['wrong']} ({stats['wrong'] / len(sample_indices) * 100:.1f}%)"
        )
        print(f"{'=' * 60}\n")

        # Apply corrections to ALL commands (extrapolate patterns)
        if corrections:
            corrected_commands = self._apply_corrections_with_heuristics(
                commands, corrections, sample_indices
            )

            # Save corrected version
            output_path = os.path.join(
                self.cfg.output_dir, f"{split_name}_commands_qc.json"
            )
            with open(output_path, "w") as f:
                json.dump(corrected_commands, f, indent=2)

            print(f"✓ Saved corrected labels to {output_path}")
            return corrections
        else:
            print("✓ No corrections needed! Labels look good.")
            return {}

    def _suggest_fix(self, reason, objects):
        """Suggest correct label based on VLM reason"""
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
        """
        Apply corrections + extrapolate to similar cases
        If VLM found specific patterns wrong, fix them everywhere
        """
        corrected = commands.copy()

        # Direct corrections from validated samples
        for idx, correction in corrections.items():
            corrected[idx]["a"] = correction["new_label"]
            corrected[idx]["qc_corrected"] = True
            corrected[idx]["qc_reason"] = correction["reason"]

        # Heuristic extrapolation (optional - be conservative)
        # If VLM found a specific pattern (e.g., "safe when red light present")
        # you could apply that fix to non-sampled data too
        # For now, just apply direct corrections

        print(f"Applied {len(corrections)} direct corrections")

        return corrected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.10,
        help="% of dataset to validate (0.10 = 10%)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to validate",
    )
    args = parser.parse_args()

    qc = PreTrainingQC(sample_rate=args.sample_rate)

    for split in args.splits:
        qc.validate_split(split)

    print("\n" + "=" * 60)
    print("PRE-TRAINING QC COMPLETE")
    print("=" * 60)
    print("\nYou can now train with corrected labels:")
    print("  python main.py --experiment_name clean_labels_v1 --epochs 15")
