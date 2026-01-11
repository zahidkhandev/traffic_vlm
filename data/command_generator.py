import json
import os
import random
from typing import Any, Dict, List

import h5py
import matplotlib.path as mplPath
import numpy as np
from tqdm import tqdm

from config.dataset_config import DatasetConfig


class CommandGenerator:
    """
    Generates VLM commands.

    UPDATES:
    - Added logic to detect 'barrier', 'traffic cone', etc. as 'stop_obstacle'.
    """

    def __init__(self):
        self.cfg = DatasetConfig()
        self.image_width = 1280
        self.image_height = 720

        self.horizon_y = self.image_height * 0.45

        # Trapezoidal Lane Logic
        self.roi_polygon = [
            (self.image_width * 0.40, self.horizon_y),
            (self.image_width * 0.60, self.horizon_y),
            (self.image_width * 0.95, self.image_height),
            (self.image_width * 0.05, self.image_height),
        ]
        self.lane_path = mplPath.Path(self.roi_polygon)

    def _point_in_lane(self, x, y):
        return self.lane_path.contains_point((x, y))

    def _is_relevant(self, box, category):
        foot_x = (box["x1"] + box["x2"]) / 2
        foot_y = box["y2"]
        box_h = box["y2"] - box["y1"]
        box_area = (box["x2"] - box["x1"]) * box_h
        img_area = self.image_width * self.image_height
        coverage = box_area / img_area

        if category == "traffic light":
            is_large_enough = (box_h / self.image_height) > 0.025
            is_centered = (self.image_width * 0.20) < foot_x < (self.image_width * 0.80)
            is_high_up = foot_y < (self.image_height * 0.65)
            return is_large_enough and is_centered and is_high_up

        if foot_y < self.horizon_y:
            return False
        in_corridor = self._point_in_lane(foot_x, foot_y)
        is_huge = coverage > 0.05
        return in_corridor or is_huge

    def _generate_safety(self, objects) -> List[Dict[str, Any]]:
        """Prioritizes stop reasons."""
        reasons = []

        for obj in objects:
            if "box2d" not in obj:
                continue

            cat = obj["category"]
            attr = obj.get("attributes", {})
            box = obj["box2d"]

            # Occlusion Check
            is_occluded = attr.get("occluded", False)
            box_h = box["y2"] - box["y1"]
            if is_occluded and cat != "traffic light":
                if (box_h / self.image_height) < 0.10:
                    continue

            # Relevance Check
            if not self._is_relevant(box, cat):
                continue

            # --- UPDATED CATEGORY LOGIC ---
            if cat == "traffic light":
                if attr.get("trafficLightColor") in ["red", "yellow"]:
                    reasons.append("stop_red_light")

            elif cat in ["pedestrian", "person", "rider"]:
                reasons.append("stop_pedestrian")

            elif cat in ["car", "bus", "truck", "train", "motor", "bike", "motorcycle"]:
                if attr.get("state") != "parked":
                    reasons.append("stop_vehicle")

            # --- NEW: OBSTACLE DETECTION ---
            elif cat in ["barrier", "traffic cone", "traffic sign", "trailer", "other"]:
                # If a barrier or cone is IN THE LANE (checked by _is_relevant), it's a stop.
                reasons.append("stop_obstacle")

        # --- UPDATED PRIORITY RESOLUTION ---
        if "stop_red_light" in reasons:
            final_a = "stop_red_light"
        elif "stop_pedestrian" in reasons:
            final_a = "stop_pedestrian"
        elif "stop_vehicle" in reasons:
            final_a = "stop_vehicle"
        elif "stop_obstacle" in reasons:
            final_a = "stop_obstacle"
        else:
            final_a = "safe"

        return [{"q": "Can I move forward?", "a": final_a, "type": "safety"}]

    def generate_commands_for_split(self, split_name: str):
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        if not os.path.exists(h5_path):
            print(f"H5 file not found: {h5_path}")
            return

        print(f"Generating SAFETY commands for {split_name}...")
        raw_cmds = []

        with h5py.File(h5_path, "r") as hf:
            meta_ds = hf["metadata"]
            if not isinstance(meta_ds, h5py.Dataset):
                return

            for idx in tqdm(range(len(meta_ds)), desc=f"Scanning {split_name}"):
                raw_data = meta_ds[idx]
                json_str = (
                    raw_data.decode("utf-8")
                    if isinstance(raw_data, bytes)
                    else str(raw_data)
                )
                meta = json.loads(json_str)
                objs = meta["frames"][0].get("objects", []) if "frames" in meta else []

                frame_cmds = self._generate_safety(objs)

                for c in frame_cmds:
                    c["image_idx"] = idx
                    raw_cmds.append(c)

        if not raw_cmds:
            return

        random.shuffle(raw_cmds)

        out_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        with open(out_path, "w") as f:
            json.dump(raw_cmds, f, indent=2)

        print(f"Successfully saved {len(raw_cmds)} commands to {out_path}")


if __name__ == "__main__":
    gen = CommandGenerator()
    for s in ["train", "val", "test"]:
        gen.generate_commands_for_split(s)
