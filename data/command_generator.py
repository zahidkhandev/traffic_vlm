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

        # ALL 10 BDD100K Object Classes
        self.valid_objects = [
            "car",
            "bus",
            "truck",
            "train",  # Vehicles
            "motor",
            "bike",  # Two-wheelers
            "pedestrian",
            "rider",  # People
            "traffic light",
            "traffic sign",  # Signals
        ]

    def _is_center(self, box):
        x1, x2 = box["x1"], box["x2"]
        center_x = (x1 + x2) / 2
        # Check if object is in the middle third of the image
        return (self.image_width / 3) < center_x < (2 * self.image_width / 3)

    def _generate_detection(self, objects):
        """Generates 'Is there a [obj]?' questions for all 10 classes."""
        present_categories = set(obj["category"] for obj in objects)
        commands = []

        # 1. Positive Sample (It exists)
        # We try to generate up to 2 positive questions to cover more objects
        if present_categories:
            valid_present = [c for c in present_categories if c in self.valid_objects]
            if valid_present:
                # Pick up to 2 distinct objects if available
                targets = random.sample(valid_present, min(len(valid_present), 2))
                for target in targets:
                    commands.append(
                        {"q": f"Is there a {target}?", "a": "yes", "type": "detection"}
                    )

        # 2. Negative Sample (It does not exist)
        missing_categories = [
            c for c in self.valid_objects if c not in present_categories
        ]
        if missing_categories:
            target = random.choice(missing_categories)
            commands.append(
                {"q": f"Is there a {target}?", "a": "no", "type": "detection"}
            )

        return commands

    def _generate_safety(self, objects):
        """
        Determines safety based on Critical Objects in the Center Lane.
        Unsafe if: Red Light OR Pedestrian OR Rider is directly ahead.
        """
        safe = True

        for obj in objects:
            # Rule 1: Red Traffic Light in Center
            if obj["category"] == "traffic light":
                color = obj.get("attributes", {}).get("trafficLightColor", "none")
                if color == "red" and self._is_center(obj["box2d"]):
                    safe = False
                    break

            # Rule 2: Vulnerable Road Users in Center (Pedestrian OR Rider)
            if obj["category"] in ["pedestrian", "rider"] and self._is_center(
                obj["box2d"]
            ):
                safe = False
                break

        commands = []
        commands.append(
            {"q": "Can I move forward?", "a": "yes" if safe else "no", "type": "safety"}
        )

        return commands

    def _generate_color(self, objects):
        """Generates questions about traffic light colors."""
        commands = []
        lights = [o for o in objects if o["category"] == "traffic light"]

        for light in lights:
            color = light.get("attributes", {}).get("trafficLightColor", "none")
            # Only ask if the color is actually defined
            if color in ["red", "green", "yellow"]:
                commands.append(
                    {"q": f"Is the traffic light {color}?", "a": "yes", "type": "state"}
                )

                # Generate a negative pair
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

    def generate_commands_for_split(self, split_name):
        h5_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            return

        print(f"Generating commands for {split_name}...")

        dataset_commands = []

        with h5py.File(h5_path, "r") as hf:
            # Explicitly access the dataset
            meta_ds = hf["metadata"]

            if isinstance(meta_ds, h5py.Dataset):
                num_samples = len(meta_ds)

                for idx in tqdm(range(num_samples)):
                    meta_str = meta_ds[idx].decode("utf-8")
                    meta = json.loads(meta_str)

                    # Handle BDD100K 'frames' structure
                    if "frames" in meta and len(meta["frames"]) > 0:
                        objects = meta["frames"][0]["objects"]
                    else:
                        objects = []

                    image_cmds = []
                    image_cmds.extend(self._generate_detection(objects))
                    image_cmds.extend(self._generate_safety(objects))
                    image_cmds.extend(self._generate_color(objects))

                    for cmd in image_cmds:
                        cmd["image_idx"] = idx
                        dataset_commands.append(cmd)
            else:
                print("Error: 'metadata' is not a valid Dataset.")

        out_path = os.path.join(self.cfg.output_dir, f"{split_name}_commands.json")
        with open(out_path, "w") as f:
            json.dump(dataset_commands, f, indent=2)

        print(f"Saved {len(dataset_commands)} commands to {out_path}")


if __name__ == "__main__":
    gen = CommandGenerator()
    gen.generate_commands_for_split("train")
    gen.generate_commands_for_split("val")
    gen.generate_commands_for_split("test")
