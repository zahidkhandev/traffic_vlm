import json
import os
import random

import h5py
from tqdm import tqdm

from config.dataset_config import DatasetConfig


class CommandGenerator:
    """
    Generates linguistically diverse and spatially accurate VLM commands.
    Implements priority-based safety logic: Red Light > Pedestrian > Vehicle.
    """

    def __init__(self):
        self.cfg = DatasetConfig()
        self.image_width = 1280
        self.image_height = 720

        # PRODUCTION SPATIAL BOTTLENECK
        # We only care about objects in the 'ego-lane' (approx 30% center width)
        self.ego_left = self.image_width * 0.35
        self.ego_right = self.image_width * 0.65

        # HORIZON THRESHOLD
        # Objects above 45% of the image height are too far to require stopping.
        self.horizon_y = self.image_height * 0.45

    def _is_relevant(self, box, category):
        """Determines if an object is physically in the path and close enough to matter."""
        center_x = (box["x1"] + box["x2"]) / 2
        bottom_y = box["y2"]
        box_h = box["y2"] - box["y1"]
        height_ratio = box_h / self.image_height

        # 1. Horizontal path check (Ego-lane)
        in_path = self.ego_left < center_x < self.ego_right

        # 2. Distance check (Horizon & Height)
        # Objects at the horizon or extremely small are ignored for safety labels
        is_close = bottom_y > self.horizon_y and height_ratio > 0.04

        # Traffic lights have a wider relevance zone but strict vertical height requirement
        if category == "traffic light":
            # Traffic lights can be slightly off-center (above the lane)
            light_relevant = (
                (self.image_width * 0.25) < center_x < (self.image_width * 0.75)
            )
            return light_relevant and height_ratio > 0.02

        return in_path and is_close

    def _generate_safety(self, objects):
        """Prioritizes stop reasons to prevent 'noisy' overlapping labels."""
        reasons = []

        for obj in objects:
            if "box2d" not in obj:
                continue
            cat = obj["category"]
            attr = obj.get("attributes", {})

            # Filter out occluded or irrelevant objects
            if attr.get("occluded") is True and cat != "traffic light":
                continue
            if not self._is_relevant(obj["box2d"], cat):
                continue

            # Logic check by priority
            if cat == "traffic light":
                if attr.get("trafficLightColor") == "red":
                    reasons.append("stop_red_light")
            elif cat in ["pedestrian", "person", "rider"]:
                reasons.append("stop_pedestrian")
            elif cat in ["car", "bus", "truck", "train", "motor", "bike"]:
                reasons.append("stop_vehicle")

        # PRIORITY RESOLUTION: We only pick one definitive reason
        if "stop_red_light" in reasons:
            final_a = "stop_red_light"
        elif "stop_pedestrian" in reasons:
            final_a = "stop_pedestrian"
        elif "stop_vehicle" in reasons:
            final_a = "stop_vehicle"
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

        target_count = max(len(g) for g in groups.values())
        print(f"   Balancing Safety Classes (Target: {target_count}):")

        balanced = []
        for ans, group in groups.items():
            print(f"     - {ans}: {len(group)} -> {target_count}")
            factor = target_count // len(group)
            balanced.extend((group * factor) + group[: target_count % len(group)])

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
            # 1. Access the dataset
            meta_ds = hf["metadata"]

            # 2. TYPE NARROWING: Explicitly verify this is an h5py Dataset for Pylance
            if not isinstance(meta_ds, h5py.Dataset):
                print(f"Error: 'metadata' in {h5_path} is not a Dataset.")
                return

            # 3. Iterate
            for idx in tqdm(range(len(meta_ds)), desc=f"Scanning {split_name}"):
                # 4. Fetch raw data safely
                raw_data = meta_ds[idx]

                # 5. Handle type safely (bytes vs str)
                if isinstance(raw_data, bytes):
                    json_str = raw_data.decode("utf-8")
                else:
                    json_str = str(raw_data)

                meta = json.loads(json_str)

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
