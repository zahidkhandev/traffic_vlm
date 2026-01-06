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
        """Generates 'Is there a [obj]?' questions."""
        present_categories = set(obj["category"] for obj in objects)
        commands = []

        # 1. Positive Sample (It exists)
        if present_categories:
            valid_present = [c for c in present_categories if c in self.valid_objects]
            if valid_present:
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

    def _generate_context(self, attributes):
        """
        Generates questions about Weather and Time of Day.
        """
        commands = []
        weather = attributes.get("weather", "undefined")
        time_of_day = attributes.get("timeofday", "undefined")

        # --- Weather Questions ---
        if weather != "undefined":
            commands.append({"q": f"Is it {weather}?", "a": "yes", "type": "weather"})
            # Simple contrast: if rainy -> clear? if clear -> rainy?
            contrast = "rainy" if weather == "clear" else "clear"
            commands.append({"q": f"Is it {contrast}?", "a": "no", "type": "weather"})

        # --- Time Questions ---
        if time_of_day != "undefined":
            commands.append({"q": f"Is it {time_of_day}?", "a": "yes", "type": "time"})
            contrast = "night" if time_of_day == "daytime" else "daytime"
            commands.append({"q": f"Is it {contrast}?", "a": "no", "type": "time"})

        return commands

    def _generate_safety(self, objects):
        """
        Determines safety based on Critical Objects in the Center Lane.
        """
        safe = True

        for obj in objects:
            # Rule 1: Red Traffic Light in Center
            if obj["category"] == "traffic light":
                color = obj.get("attributes", {}).get("trafficLightColor", "none")
                if color == "red" and self._is_center(obj["box2d"]):
                    safe = False
                    break

            # Rule 2: Vulnerable Road Users in Center
            if obj["category"] in ["pedestrian", "rider"] and self._is_center(
                obj["box2d"]
            ):
                safe = False
                break

        return [
            {"q": "Can I move forward?", "a": "yes" if safe else "no", "type": "safety"}
        ]

    def _generate_color(self, objects):
        """Generates questions about traffic light colors."""
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
                break  # Only ask about one light per image
        return commands

    def _balance_safety_data(self, commands):
        """
        Balances Safety data by UPSAMPLING (Duplicating) the minority class.
        This ensures the model sees equal Yes/No examples without losing data.
        """
        # Separate safety questions from the rest
        safety_yes = [c for c in commands if c["type"] == "safety" and c["a"] == "yes"]
        safety_no = [c for c in commands if c["type"] == "safety" and c["a"] == "no"]
        others = [c for c in commands if c["type"] != "safety"]

        if not safety_yes or not safety_no:
            print("Warning: Could not balance safety data (one class is empty).")
            return commands

        # Target is the MAJORITY count (we want to match the bigger one)
        target_count = max(len(safety_yes), len(safety_no))

        print(f"   Original Safety: Yes={len(safety_yes)}, No={len(safety_no)}")
        print(f"   Upsampling to match: {target_count}")

        # Upsample the minority 'Yes'
        if len(safety_yes) < target_count:
            factor = target_count // len(safety_yes)
            remainder = target_count % len(safety_yes)
            safety_yes = (safety_yes * factor) + safety_yes[:remainder]

        # Upsample the minority 'No'
        if len(safety_no) < target_count:
            factor = target_count // len(safety_no)
            remainder = target_count % len(safety_no)
            safety_no = (safety_no * factor) + safety_no[:remainder]

        print(f"   Final Safety: Yes={len(safety_yes)}, No={len(safety_no)}")

        # Combine everything back
        final_commands = others + safety_yes + safety_no
        random.shuffle(final_commands)  # Shuffle so training isn't clustered

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
                # Iterate through every image in the H5 file
                for idx in tqdm(range(len(meta_ds))):
                    meta_str = meta_ds[idx].decode("utf-8")
                    meta = json.loads(meta_str)

                    # Safely extract objects and attributes
                    if "frames" in meta and len(meta["frames"]) > 0:
                        objects = meta["frames"][0]["objects"]
                    else:
                        objects = []

                    attributes = meta.get("attributes", {})

                    # Generate all question types
                    image_cmds = []
                    image_cmds.extend(self._generate_detection(objects))
                    image_cmds.extend(self._generate_safety(objects))
                    image_cmds.extend(self._generate_color(objects))
                    image_cmds.extend(self._generate_context(attributes))

                    # Tag them with the image index
                    for cmd in image_cmds:
                        cmd["image_idx"] = idx
                        raw_commands.append(cmd)
            else:
                print("Error: 'metadata' is not a valid Dataset.")

        # --- APPLY UPSAMPLING HERE ---
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
