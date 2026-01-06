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

        # All BDD100K object classes
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

        # Context attributes
        self.context_attributes = [
            "weather",
            "timeofday",
            "sky",
            "illumination",
            "precipitation",
            "infrastructure",
            "road",
            "tunnel",
            "construction_site",
            "clear_windshield",
            "light_exposure",
            "reflections",
        ]

    def _is_center(self, box):
        x1, x2 = box["x1"], box["x2"]
        center_x = (x1 + x2) / 2
        return (self.image_width / 3) < center_x < (2 * self.image_width / 3)

    def _is_close(self, box, depth_info=None, threshold=0.7):
        # Estimate proximity using bounding box size or depth if available
        box_width = box["x2"] - box["x1"]
        if depth_info:
            # Use depth if available
            return depth_info.get(box.get("id"), 1.0) < threshold
        else:
            # Use box size as proxy for distance
            return box_width / self.image_width > threshold

    def _generate_detection(self, objects):
        present_categories = set(obj["category"] for obj in objects)
        commands = []

        # Positive samples
        if present_categories:
            valid_present = [c for c in present_categories if c in self.valid_objects]
            if valid_present:
                targets = random.sample(valid_present, min(len(valid_present), 2))
                for target in targets:
                    commands.append(
                        {"q": f"Is there a {target}?", "a": "yes", "type": "detection"}
                    )

        # Negative samples
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
            value = attributes.get(attr, "undefined")
            if value != "undefined":
                commands.append({"q": f"Is it {value}?", "a": "yes", "type": "context"})
                # Add contrast question for binary attributes
                if attr == "timeofday":
                    contrast = "night" if value == "daytime" else "daytime"
                    commands.append(
                        {"q": f"Is it {contrast}?", "a": "no", "type": "context"}
                    )
                elif attr == "weather":
                    contrast = "rainy" if value == "clear" else "clear"
                    commands.append(
                        {"q": f"Is it {contrast}?", "a": "no", "type": "context"}
                    )
        return commands

    def _generate_safety(self, objects, depth_info=None):
        safe = True
        for obj in objects:
            # Red traffic light in center and close
            if obj["category"] == "traffic light":
                color = obj.get("attributes", {}).get("trafficLightColor", "none")
                if (
                    color == "red"
                    and self._is_center(obj["box2d"])
                    and self._is_close(obj["box2d"], depth_info)
                ):
                    safe = False
                    break

            # Vulnerable road users (pedestrian, rider) in center and close
            if (
                obj["category"] in ["pedestrian", "rider"]
                and self._is_center(obj["box2d"])
                and self._is_close(obj["box2d"], depth_info)
            ):
                safe = False
                break

            # Large vehicles (bus, truck) in center and close
            if (
                obj["category"] in ["bus", "truck"]
                and self._is_center(obj["box2d"])
                and self._is_close(obj["box2d"], depth_info)
            ):
                safe = False
                break

        return [
            {"q": "Can I move forward?", "a": "yes" if safe else "no", "type": "safety"}
        ]

    def _generate_color(self, objects):
        commands = []
        lights = [o for o in objects if o["category"] == "traffic light"]
        for light in lights:
            color = light.get("attributes", {}).get("trafficLightColor", "none")
            if color in ["red", "green", "yellow"]:
                commands.append(
                    {"q": f"Is the traffic light {color}?", "a": "yes", "type": "state"}
                )
                fake_color = "red" if color == "green" else "green"
                commands.append(
                    {
                        "q": f"Is the traffic light {fake_color}?",
                        "a": "no",
                        "type": "state",
                    }
                )
                break
        return commands

    def _balance_safety_data(self, commands):
        safety_yes = [c for c in commands if c["type"] == "safety" and c["a"] == "yes"]
        safety_no = [c for c in commands if c["type"] == "safety" and c["a"] == "no"]
        others = [c for c in commands if c["type"] != "safety"]

        if not safety_yes or not safety_no:
            print("Warning: Could not balance safety data (one class is empty).")
            return commands

        target_count = max(len(safety_yes), len(safety_no))
        print(f"   Original Safety: Yes={len(safety_yes)}, No={len(safety_no)}")
        print(f"   Upsampling to match: {target_count}")

        if len(safety_yes) < target_count:
            factor = target_count // len(safety_yes)
            remainder = target_count % len(safety_yes)
            safety_yes = (safety_yes * factor) + safety_yes[:remainder]

        if len(safety_no) < target_count:
            factor = target_count // len(safety_no)
            remainder = target_count % len(safety_no)
            safety_no = (safety_no * factor) + safety_no[:remainder]

        print(f"   Final Safety: Yes={len(safety_yes)}, No={len(safety_no)}")

        final_commands = others + safety_yes + safety_no
        random.shuffle(final_commands)
        return final_commands

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
                    meta_str = meta_ds[idx].decode("utf-8")
                    meta = json.loads(meta_str)

                    if "frames" in meta and len(meta["frames"]) > 0:
                        objects = meta["frames"][0]["objects"]
                    else:
                        objects = []

                    attributes = meta.get("attributes", {})

                    # Extract depth info if available
                    depth_info = {}
                    if "frames" in meta and "depth" in meta["frames"][0]:
                        for obj in objects:
                            depth_info[obj["id"]] = meta["frames"][0]["depth"].get(
                                obj["id"], 1.0
                            )

                    image_cmds = []
                    image_cmds.extend(self._generate_detection(objects))
                    image_cmds.extend(self._generate_safety(objects, depth_info))
                    image_cmds.extend(self._generate_color(objects))
                    image_cmds.extend(self._generate_context(attributes))

                    for cmd in image_cmds:
                        cmd["image_idx"] = idx
                        raw_commands.append(cmd)
            else:
                print("Error: 'metadata' is not a valid Dataset.")

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
