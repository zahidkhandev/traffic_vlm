import json
import os
import random

import h5py
import matplotlib.path as mplPath
import numpy as np
from tqdm import tqdm

from config.dataset_config import DatasetConfig


class CommandGenerator:
    """
    Generates linguistically diverse and spatially accurate VLM commands.

    MAJOR UPDATE:
    - Replaced rectangular crop with Trapezoidal Lane Logic (ROI).
    - Uses 'feet' position (bottom-center) of boxes to determine distance/lane.
    - Improved occlusion logic (ignores occlusion if object is very close/dangerous).
    """

    def __init__(self):
        self.cfg = DatasetConfig()
        self.image_width = 1280
        self.image_height = 720

        # --- SPATIAL CONFIGURATION ---

        # 1. Horizon Line: Objects above this Y-value are too far away.
        self.horizon_y = self.image_height * 0.45

        # 2. Driving Corridor (Trapezoid)
        # We define a polygon that mimics the perspective of a road lane.
        # Top points (Horizon) are narrow; Bottom points (Hood) are wide.
        self.roi_polygon = [
            (self.image_width * 0.40, self.horizon_y),  # Top-Left
            (self.image_width * 0.60, self.horizon_y),  # Top-Right
            (self.image_width * 0.95, self.image_height),  # Bottom-Right (Hood)
            (self.image_width * 0.05, self.image_height),  # Bottom-Left (Hood)
        ]
        self.lane_path = mplPath.Path(self.roi_polygon)

    def _point_in_lane(self, x, y):
        """Checks if a specific point (x, y) is inside the driving corridor."""
        return self.lane_path.contains_point((x, y))

    def _is_relevant(self, box, category):
        """
        Determines if an object is relevant based on road geometry.
        Uses the 'feet' (bottom-center) of the object for ground placement.
        """
        # 1. Calculate Key Metrics
        foot_x = (box["x1"] + box["x2"]) / 2
        foot_y = box["y2"]  # The point where the object touches the road

        box_h = box["y2"] - box["y1"]
        box_area = (box["x2"] - box["x1"]) * box_h
        img_area = self.image_width * self.image_height
        coverage = box_area / img_area

        # 2. Traffic Light Logic (Special Case)
        # Lights are not "on the road", they are in the air.
        if category == "traffic light":
            # Must be relatively large/close OR very bright/central
            is_large_enough = (box_h / self.image_height) > 0.025

            # Widen the x-check for lights (often on poles to the side)
            is_centered = (self.image_width * 0.20) < foot_x < (self.image_width * 0.80)

            # Must be in the upper 2/3rds of the image (lights aren't on the floor)
            is_high_up = foot_y < (self.image_height * 0.65)

            return is_large_enough and is_centered and is_high_up

        # 3. Ground Object Logic (Cars, Peds)

        # A. Horizon Check: Is it too far away?
        if foot_y < self.horizon_y:
            return False

        # B. Geometry Check: Are the feet inside our lane trapezoid?
        in_corridor = self._point_in_lane(foot_x, foot_y)

        # C. Danger Check: Is it huge?
        # If an object covers >5% of the screen, it's relevant regardless of alignment
        # (e.g., a car cutting in from the side close up).
        is_huge = coverage > 0.05

        return in_corridor or is_huge

    def _generate_safety(self, objects):
        """Prioritizes stop reasons with improved occlusion handling."""
        reasons = []

        for obj in objects:
            if "box2d" not in obj:
                continue

            cat = obj["category"]
            attr = obj.get("attributes", {})
            box = obj["box2d"]

            # --- OCCLUSION FILTERING ---
            is_occluded = attr.get("occluded", False)
            box_h = box["y2"] - box["y1"]

            # If occluded, only ignore it if it's ALSO small/far.
            # If a pedestrian is occluded but huge (close), we MUST stop.
            if is_occluded and cat != "traffic light":
                if (box_h / self.image_height) < 0.10:
                    continue

            # --- RELEVANCE FILTERING ---
            if not self._is_relevant(box, cat):
                continue

            # --- CATEGORY LOGIC ---
            if cat == "traffic light":
                # Stop for Red or Yellow (Yellow implies caution/stop usually)
                if attr.get("trafficLightColor") in ["red", "yellow"]:
                    reasons.append("stop_red_light")

            elif cat in ["pedestrian", "person", "rider"]:
                reasons.append("stop_pedestrian")

            elif cat in ["car", "bus", "truck", "train", "motor", "bike"]:
                # (Optional) Check 'parked' attribute here if your dataset has it
                if attr.get("state") != "parked":
                    reasons.append("stop_vehicle")

        # --- PRIORITY RESOLUTION ---
        # 1. Red Light (Legal imperative)
        if "stop_red_light" in reasons:
            final_a = "stop_red_light"
        # 2. Pedestrian (Human safety)
        elif "stop_pedestrian" in reasons:
            final_a = "stop_pedestrian"
        # 3. Vehicle (Traffic flow)
        elif "stop_vehicle" in reasons:
            final_a = "stop_vehicle"
        # 4. Safe
        else:
            final_a = "safe"

        return [{"q": "Can I move forward?", "a": final_a, "type": "safety"}]

    def _generate_detection(self, objects):
        """Generates yes/no questions based on visible, non-occluded objects."""
        visible_cats = {
            o["category"] for o in objects if not o.get("attributes", {}).get("occluded")
        }
        if "person" in visible_cats:
            visible_cats.add("pedestrian")

        targets = ["car", "pedestrian", "traffic light", "truck", "bus"]
        cmds = []
        for t in targets:
            ans = "yes" if t in visible_cats else "no"
            cmds.append({"q": f"Is there a {t}?", "a": ans, "type": "detection"})
        return cmds

    def _generate_context(self, attributes):
        """Encodes weather and time of day."""
        cmds = []
        for attr in ["weather", "timeofday"]:
            val = attributes.get(attr, "undefined")
            if val != "undefined":
                cmds.append({"q": f"What is the {attr}?", "a": val, "type": "context"})
        return cmds

    def _balance_safety_data(self, commands):
        """Balances safety classes to ensure rare events (pedestrians) are seen."""
        safety = [c for c in commands if c["type"] == "safety"]
        others = [c for c in commands if c["type"] != "safety"]

        if not safety:
            return commands

        groups = {}
        for c in safety:
            groups[c["a"]] = groups.get(c["a"], []) + [c]

        # Avoid empty sequence error if groups is empty
        if not groups:
            return commands

        target_count = max(len(g) for g in groups.values())
        print(f"   Balancing Safety Classes (Target: {target_count}):")

        balanced = []
        for ans, group in groups.items():
            count = len(group)
            print(f"     - {ans}: {count} -> {target_count}")
            if count == 0:
                continue

            factor = target_count // count
            remainder = target_count % count
            balanced.extend((group * factor) + group[:remainder])

        final = balanced + others
        random.shuffle(final)
        return final

    def generate_commands_for_split(self, split_name: str):
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        if not os.path.exists(h5_path):
            print(f"H5 file not found: {h5_path}")
            return

        print(f"Generating commands for {split_name}...")
        raw_cmds = []

        with h5py.File(h5_path, "r") as hf:
            meta_ds = hf["metadata"]

            if not isinstance(meta_ds, h5py.Dataset):
                print(f"Error: 'metadata' in {h5_path} is not a Dataset.")
                return

            # Use tqdm for progress tracking
            for idx in tqdm(range(len(meta_ds)), desc=f"Scanning {split_name}"):
                raw_data = meta_ds[idx]

                if isinstance(raw_data, bytes):
                    json_str = raw_data.decode("utf-8")
                else:
                    json_str = str(raw_data)

                meta = json.loads(json_str)

                # Safe access to objects and attributes
                objs = meta["frames"][0].get("objects", []) if "frames" in meta else []
                attrs = meta.get("attributes", {})

                frame_cmds = []
                frame_cmds.extend(self._generate_safety(objs))
                frame_cmds.extend(self._generate_detection(objs))
                frame_cmds.extend(self._generate_context(attrs))

                for c in frame_cmds:
                    c["image_idx"] = idx
                    raw_cmds.append(c)

        if not raw_cmds:
            print(f"No commands generated for {split_name}.")
            return

        final = self._balance_safety_data(raw_cmds)
        out_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)

        print(f"Successfully saved {len(final)} commands to {out_path}")


if __name__ == "__main__":
    gen = CommandGenerator()
    for s in ["train", "val", "test"]:
        gen.generate_commands_for_split(s)
