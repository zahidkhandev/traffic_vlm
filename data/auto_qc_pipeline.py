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
    def __init__(self, use_vlm=True):
        self.cfg = DatasetConfig()
        self.cleanlab = CleanLabQC()

        local_model_path = r"C:\Users\nkz3kor\Documents\traffic_vlm\llava-model"
        self.vlm_judge = VLMJudge(model_name=local_model_path) if use_vlm else None

    def run_qc(self, split_name, model_predictions=None):
        print(f"\n=== Starting AutoQC for {split_name} ===")

        cmd_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")

        with open(cmd_path, "r") as f:
            commands = json.load(f)

        label_map = self.cfg.label_map
        labels = [label_map[cmd["a"].lower().strip()] for cmd in commands]

        suspicious_indices = []
        if model_predictions is not None:
            print("\n[Layer 3] Running Cleanlab QC...")
            suspicious_indices = self.cleanlab.find_errors(labels, model_predictions)
        else:
            print("[Layer 3] No model predictions. Sampling 5% for VLM check...")
            n_samples = int(len(commands) * 0.05)
            suspicious_indices = np.random.choice(len(commands), n_samples, replace=False)

        if self.vlm_judge and len(suspicious_indices) > 0:
            print(
                f"\n[Layer 4] VLM validating {len(suspicious_indices)} suspicious samples..."
            )
            corrections = self._vlm_validate(h5_path, commands, suspicious_indices)

            corrected_commands = self._apply_corrections(commands, corrections)

            corrected_path = os.path.join(
                self.cfg.output_dir, f"{split_name}_commands_qc.json"
            )
            with open(corrected_path, "w") as f:
                json.dump(corrected_commands, f, indent=2)

            print(f"Corrected {len(corrections)} labels")
            print(f"Saved to {corrected_path}")

            return corrected_commands, corrections

        return commands, {}

    def _vlm_validate(self, h5_path, commands, indices):
        corrections = {}

        with h5py.File(h5_path, "r") as hf:
            images_ds = hf["images"]
            metadata_ds = hf["metadata"]

            if not isinstance(images_ds, h5py.Dataset) or not isinstance(
                metadata_ds, h5py.Dataset
            ):
                raise TypeError("H5 datasets not loaded correctly")

            for idx in tqdm(indices[:100], desc="VLM Validation"):
                cmd = commands[idx]
                img_idx = cmd["image_idx"]

                img_array = np.array(images_ds[img_idx])
                image = Image.fromarray(img_array.astype("uint8"))

                meta_bytes = metadata_ds[img_idx]
                if isinstance(meta_bytes, bytes):
                    meta_str = meta_bytes.decode("utf-8")
                elif isinstance(meta_bytes, np.bytes_):
                    meta_str = meta_bytes.tobytes().decode("utf-8")
                else:
                    meta_str = str(meta_bytes)

                meta = json.loads(meta_str)
                objects = meta["frames"][0].get("objects", [])

                if self.vlm_judge is None:
                    continue

                is_valid, confidence, reason = self.vlm_judge.validate_label(
                    image, objects, cmd["a"]
                )

                if not is_valid:
                    suggested_fix = self._suggest_fix(reason, objects)

                    if suggested_fix == cmd["a"]:
                        continue

                    corrections[idx] = {
                        "old_label": cmd["a"],
                        "confidence": confidence,
                        "reason": reason,
                        "suggested_fix": suggested_fix,
                    }

        return corrections

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

    def _apply_corrections(self, commands, corrections):
        corrected = commands.copy()

        for idx, correction in corrections.items():
            if "suggested_fix" in correction:
                corrected[idx]["a"] = correction["suggested_fix"]
                corrected[idx]["qc_corrected"] = True
                corrected[idx]["qc_reason"] = correction["reason"]

        return corrected
