import json
import os
from collections import Counter

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from config.dataset_config import DatasetConfig


class DatasetBuilder:
    """
    Handles the processing of raw image and label data into structured HDF5 files.

    This class scans directory structures, filters samples based on the provided
    configuration (weather, time of day), resizes images, and aggregates them
    into a single .h5 file for efficient training access.
    """

    def __init__(self):
        """
        Initialize the DatasetBuilder with the default configuration.
        """
        self.cfg = DatasetConfig()

    def process_split(self, split_name: str):
        """
        Process a specific data split (e.g., 'train', 'val', 'test').

        Scans the corresponding directory for valid samples that match the
        configuration criteria, prints distribution statistics, and saves
        the data to an HDF5 file.

        Args:
            split_name (str): The name of the split to process.
        """
        print(f"\n--- Processing {split_name} ---")

        img_dir = os.path.join(self.cfg.root_dir, self.cfg.images_dir_name, split_name)
        lbl_dir = os.path.join(self.cfg.root_dir, self.cfg.labels_dir_name, split_name)

        output_path = os.path.join(self.cfg.output_dir, f"{split_name}.h5")
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if not os.path.exists(lbl_dir):
            print(f"Skipping {split_name}: Label directory not found at {lbl_dir}")
            return

        json_files = [f for f in os.listdir(lbl_dir) if f.endswith(".json")]
        valid_samples = []

        weather_stats = Counter()
        time_stats = Counter()

        print(f"Scanning {len(json_files)} label files in {split_name}...")

        for j_file in tqdm(json_files, desc="Filtering Metadata"):
            full_path = os.path.join(lbl_dir, j_file)

            try:
                with open(full_path, "r") as f:
                    content = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping corrupt JSON: {j_file}")
                continue

            items = content if isinstance(content, list) else [content]

            for item in items:
                weather = item.get("attributes", {}).get("weather", "undefined")
                time_of_day = item.get("attributes", {}).get("timeofday", "undefined")

                if weather not in self.cfg.allowed_weather:
                    continue
                if time_of_day not in self.cfg.allowed_time:
                    continue

                img_name = item["name"]
                img_path = os.path.join(img_dir, img_name)

                if not os.path.exists(img_path):
                    if os.path.exists(img_path + ".jpg"):
                        img_path += ".jpg"
                    else:
                        continue

                valid_samples.append({"img_path": img_path, "json_data": item})
                weather_stats[weather] += 1
                time_stats[time_of_day] += 1

        if not valid_samples:
            print(f"No valid images found for {split_name} matching filters.")
            return

        print(f"\nFound {len(valid_samples)} valid images.")
        print("Weather Distribution:", dict(weather_stats))
        print("Time Distribution:   ", dict(time_stats))
        print(f"Saving to {output_path}...\n")

        with h5py.File(output_path, "w") as hf:
            img_ds = hf.create_dataset(
                "images",
                shape=(len(valid_samples), self.cfg.image_size, self.cfg.image_size, 3),
                dtype="uint8",
                chunks=(1, self.cfg.image_size, self.cfg.image_size, 3),
                compression="gzip",
            )

            meta_ds = hf.create_dataset(
                "metadata",
                shape=(len(valid_samples),),
                dtype=h5py.string_dtype(),
            )

            success_count = 0
            for idx, sample in enumerate(tqdm(valid_samples, desc="Writing to H5")):
                try:
                    with Image.open(sample["img_path"]) as img:
                        img = img.convert("RGB")
                        img = img.resize((self.cfg.image_size, self.cfg.image_size))
                        img_array = np.array(img)

                    img_ds[idx] = img_array
                    meta_ds[idx] = json.dumps(sample["json_data"])
                    success_count += 1

                except Exception as e:
                    print(f"Error processing {sample['img_path']}: {e}")

            print(f"Successfully wrote {success_count}/{len(valid_samples)} samples.")


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.process_split("train")
    builder.process_split("val")
    builder.process_split("test")
