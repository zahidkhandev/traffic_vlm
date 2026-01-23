import json
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from config.dataset_config import DatasetConfig
from data.cleanlab_qc import CleanLabQC
from data.vlm_qc import VLMJudge


class AutoQCPipeline:
    """
    Combines Layer 3 (Cleanlab) + Layer 4 (VLM Judge) for AutoQC
    """

    def __init__(self, use_vlm=True):
        self.cfg = DatasetConfig()
        self.cleanlab = CleanLabQC()
        self.vlm_judge = VLMJudge() if use_vlm else None

    def run_qc(self, split_name, model_predictions=None):
        """
        Main QC workflow:
        1. Load commands and H5 data
        2. Run Cleanlab if predictions available (Layer 3)
        3. Run VLM validation on suspicious samples (Layer 4)
        4. Generate corrected commands JSON
        """
        print(f"\n=== Starting AutoQC for {split_name} ===")

        cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")

        # Load commands
        with open(cmd_path, "r") as f:
            commands = json.load(f)

        # Extract labels
        label_map = self.cfg.label_map
        labels = [label_map[cmd["a"].lower().strip()] for cmd in commands]

        # Layer 3: Cleanlab (if model predictions provided)
        suspicious_indices = []
        if model_predictions is not None:
            print("\n[Layer 3] Running Cleanlab Statistical QC...")
            suspicious_indices = self.cleanlab.find_errors(labels, model_predictions)
        else:
            # If no predictions, randomly sample 5% for VLM validation
            print("[Layer 3] No model predictions. Sampling 5% for VLM check...")
            n_samples = int(len(commands) * 0.05)
            suspicious_indices = np.random.choice(len(commands), n_samples, replace=False)

        # Layer 4: VLM Judge validation
        if self.vlm_judge and len(suspicious_indices) > 0:
            print(
                f"\n[Layer 4] VLM validating {len(suspicious_indices)} suspicious samples..."
            )
            corrections = self._vlm_validate(h5_path, commands, suspicious_indices)

            # Apply corrections
            corrected_commands = self._apply_corrections(commands, corrections)

            # Save corrected version
            corrected_path = os.path.join(
                self.cfg.output_dir, f"{split_name}_commands_qc.json"
            )
            with open(corrected_path, "w") as f:
                json.dump(corrected_commands, f, indent=2)

            print(f"✓ Corrected {len(corrections)} labels")
            print(f"✓ Saved to {corrected_path}")

            return corrected_commands, corrections

        return commands, {}

    def _vlm_validate(self, h5_path, commands, indices):
        """Run VLM validation on suspicious samples"""
        corrections = {}

        with h5py.File(h5_path, "r") as hf:
            images_ds = hf["images"]
            metadata_ds = hf["metadata"]

            # Type check
            if not isinstance(images_ds, h5py.Dataset) or not isinstance(
                metadata_ds, h5py.Dataset
            ):
                raise TypeError("H5 datasets not loaded correctly")

            for idx in tqdm(indices[:100], desc="VLM Validation"):
                cmd = commands[idx]
                img_idx = cmd["image_idx"]

                # Load image
                img_array = np.array(images_ds[img_idx])
                image = Image.fromarray(img_array.astype("uint8"))

                # Load metadata for objects
                meta_bytes = metadata_ds[img_idx]
                if isinstance(meta_bytes, bytes):
                    meta_str = meta_bytes.decode("utf-8")
                elif isinstance(meta_bytes, np.bytes_):
                    meta_str = meta_bytes.tobytes().decode("utf-8")
                else:
                    meta_str = str(meta_bytes)

                meta = json.loads(meta_str)
                objects = meta["frames"][0].get("objects", [])

                # VLM judgment
                if self.vlm_judge is None:
                    continue

                is_valid, confidence, reason = self.vlm_judge.validate_label(
                    image, objects, cmd["a"]
                )

                if not is_valid:
                    corrections[idx] = {
                        "old_label": cmd["a"],
                        "confidence": confidence,
                        "reason": reason,
                        "suggested_fix": self._suggest_fix(reason, objects),
                    }

        return corrections

    def _suggest_fix(self, reason, objects):
        """Heuristic to suggest correct label based on VLM reason"""
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

    def _apply_corrections(self, commands, corrections):
        """Apply suggested fixes to commands"""
        corrected = commands.copy()

        for idx, correction in corrections.items():
            if "suggested_fix" in correction:
                corrected[idx]["a"] = correction["suggested_fix"]
                corrected[idx]["qc_corrected"] = True
                corrected[idx]["qc_reason"] = correction["reason"]

        return corrected
