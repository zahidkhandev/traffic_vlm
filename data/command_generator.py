import json
import os
import random

import h5py
from tqdm import tqdm

from config.dataset_config import DatasetConfig


class CommandGenerator:
    def __init__(self):
        self.cfg = DatasetConfig()
        self.image_width = 1280
        self.image_height = 720

        self.valid_objects = [
            "car",
            "bus",
            "truck",
            "train",
            "motor",
            "bike",
            "pedestrian",
            "rider",
            "traffic light",
            "traffic sign",
            "drivable area",
            "lane marking",
        ]

        self.context_attributes = ["weather", "timeofday", "scene", "illumination"]

    def _is_center(self, box):
        """Checks if object is in the horizontal center 50% of image."""
        x1, x2 = box["x1"], box["x2"]
        center_x = (x1 + x2) / 2
        return (self.image_width * 0.25) < center_x < (self.image_width * 0.75)

    def _is_close(self, box, category):
        """
        DETERMINES DEPTH BASED ON SIZE.
        If an object is too small, it is far away -> SAFE.
        If an object is big, it is close -> STOP.
        """
        y1, y2 = box["y1"], box["y2"]
        box_h = y2 - y1
        height_ratio = box_h / self.image_height

        # Different thresholds for different objects
        if category in ["traffic light"]:
            # Traffic lights are small, so even 5% height means it's close/relevant
            return height_ratio > 0.05
        elif category in ["pedestrian", "rider"]:
            # Humans are smaller than trucks. If a human is 8% of screen, they are close.
            return height_ratio > 0.08
        else:
            # Cars/Trucks/Buses.
            # If a car is < 10% of the screen height, it is FAR AWAY.
            # We can safely drive forward.
            return height_ratio > 0.10

    def _generate_detection(self, objects):
        present_categories = set(obj["category"] for obj in objects)
        commands = []
        if present_categories:
            valid_present = [c for c in present_categories if c in self.valid_objects]
            if valid_present:
                targets = random.sample(valid_present, min(len(valid_present), 2))
                for target in targets:
                    commands.append(
                        {"q": f"Is there a {target}?", "a": "yes", "type": "detection"}
                    )

        missing_categories = [
            c for c in self.valid_objects if c not in present_categories
        ]
        if missing_categories:
            target = random.choice(missing_categories)
            commands.append(
                {"q": f"Is there a {target}?", "a": "no", "type": "detection"}
            )
        return commands

    def _generate_context(self, attributes):
        commands = []
        for attr in self.context_attributes:
            val = attributes.get(attr, "undefined")
            if val != "undefined":
                commands.append(
                    {"q": f"Is the {attr} {val}?", "a": "yes", "type": "context"}
                )
                if attr == "weather":
                    contrast = "rainy" if val == "clear" else "clear"
                    commands.append(
                        {"q": f"Is the {attr} {contrast}?", "a": "no", "type": "context"}
                    )
        return commands

    def _generate_safety(self, objects):
        """
        Generates 'safe' OR specific reasons based on CENTER + CLOSE logic.
        """
        reason = "safe"

        for obj in objects:
            # --- FIX: Skip objects that don't have bounding boxes (like lanes) ---
            if "box2d" not in obj:
                continue
            # -------------------------------------------------------------------

            cat = obj["category"]

            # 1. Check Center
            if not self._is_center(obj["box2d"]):
                continue  # Object is on the side, safe to pass.

            # 2. Check Distance (Depth Proxy)
            if not self._is_close(obj["box2d"], cat):
                continue  # Object is center but FAR AWAY. Safe to continue.

            # If we are here, object is Center AND Close. Danger!

            # Priority 1: Red Light
            if cat == "traffic light":
                color = obj.get("attributes", {}).get("trafficLightColor", "none")
                if color == "red":
                    reason = "stop_red_light"
                    break

            # Priority 2: Humans
            elif cat in ["pedestrian", "rider"]:
                reason = "stop_pedestrian"
                break

            # Priority 3: Vehicles
            elif cat in ["car", "bus", "truck", "train", "motor", "bike"]:
                reason = "stop_vehicle"
                break

            # Priority 4: Obstacles
            elif cat not in ["drivable area", "lane marking", "traffic sign"]:
                reason = "stop_obstacle"

        return [{"q": "Can I move forward?", "a": reason, "type": "safety"}]

    def _generate_color(self, objects):
        commands = []
        lights = [o for o in objects if o["category"] == "traffic light"]
        for light in lights:
            color = light.get("attributes", {}).get("trafficLightColor", "none")
            if color in ["red", "green", "yellow"]:
                commands.append(
                    {"q": f"Is the traffic light {color}?", "a": "yes", "type": "state"}
                )
                fake = "red" if color == "green" else "green"
                commands.append(
                    {"q": f"Is the traffic light {fake}?", "a": "no", "type": "state"}
                )
                break
        return commands

    def _balance_safety_data(self, commands):
        safety_cmds = [c for c in commands if c["type"] == "safety"]
        others = [c for c in commands if c["type"] != "safety"]

        if not safety_cmds:
            return commands

        groups = {}
        for cmd in safety_cmds:
            ans = cmd["a"]
            if ans not in groups:
                groups[ans] = []
            groups[ans].append(cmd)

        max_count = max(len(g) for g in groups.values())
        print(f"   Balancing Safety Classes (Target: {max_count}):")

        balanced_safety = []
        for ans, group in groups.items():
            count = len(group)
            print(f"     - {ans}: {count} -> {max_count}")
            if count < max_count:
                if count == 0:
                    continue
                factor = max_count // count
                remainder = max_count % count
                group = (group * factor) + group[:remainder]
            balanced_safety.extend(group)

        final = others + balanced_safety
        random.shuffle(final)
        return final

    def generate_commands_for_split(self, split_name):
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            return

        print(f"Generating commands for {split_name}...")
        raw_commands = []

        with h5py.File(h5_path, "r") as hf:
            meta_ds = hf["metadata"]
            if isinstance(meta_ds, h5py.Dataset):
                for idx in tqdm(range(len(meta_ds))):
                    meta = json.loads(meta_ds[idx].decode("utf-8"))

                    objects = []
                    if "frames" in meta and len(meta["frames"]) > 0:
                        objects = meta["frames"][0].get("objects", [])

                    attributes = meta.get("attributes", {})

                    cmds = []
                    cmds.extend(self._generate_detection(objects))
                    cmds.extend(self._generate_safety(objects))
                    cmds.extend(self._generate_color(objects))
                    cmds.extend(self._generate_context(attributes))

                    for c in cmds:
                        c["image_idx"] = idx
                        raw_commands.append(c)
            else:
                print("Error reading metadata.")

        final_commands = self._balance_safety_data(raw_commands)

        out_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        with open(out_path, "w") as f:
            json.dump(final_commands, f, indent=2)
        print(f"Saved {len(final_commands)} commands to {out_path}")


if __name__ == "__main__":
    gen = CommandGenerator()
    gen.generate_commands_for_split("train")
    gen.generate_commands_for_split("val")
    gen.generate_commands_for_split("test")