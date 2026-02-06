import json
import os
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


class KMeansQC:
    def __init__(self) -> None:
        print("Loading CLIP model...")
        self.model = SentenceTransformer("clip-ViT-B-32")
        print("Model loaded successfully!")
        self.list_embeddings = []
        self.list_of_metadata = []

    def extract_embeddings(self, label_json_path, image_path):
        """
        This function opens up an image and its corrsponding labels, the label json file
        contains bounding boxes of with corresponding labels, the goal of this function
        is generate embeddings for those bounding box crops and store them in list_embeddings
        along with the image and file meta data in order
        """
        if not os.path.exists(label_json_path):
            print(f"Error: JSON file not found at {label_json_path}")
            return

        with open(label_json_path, "r") as f:
            data = json.load(f)

        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        full_img = Image.open(image_path)

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")
                box = obj.get("box2d")

                if box:
                    x1, y1 = int(box["x1"]), int(box["y1"])
                    x2, y2 = int(box["x2"]), int(box["y2"])
                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        print(f"Skipping invalid box in {image_path}: {box}")
                        self.list_of_metadata.append(
                            {
                                "category": category,
                                "file": os.path.basename(image_path),
                                "obj_id": obj.get("id"),
                                "box": [x1, y1, x2, y2],
                                "is_corrupted": True,
                                "error_msg": "Zero or Negative Area",
                            }
                        )

                        continue

                    crop = full_img.crop(box=(x1, y1, x2, y2))

                    # print(f"Cropped a {category} at [{x1}, {y1}, {x2}, {y2}]")
                    embeddings = self.model.encode([crop])  # type: ignore
                    self.list_embeddings.append(embeddings)
                    self.list_of_metadata.append(
                        {
                            "category": category,
                            "file": os.path.basename(image_path),
                            "obj_id": obj.get("id"),
                            "box": [x1, y1, x2, y2],
                            "is_corrupted": False,
                            "error_msg": "OK",
                        }
                    )

    def loop_labels(
        self, images_path, labels_path, output_dir="data/processed/k_means_qc"
    ):
        """
        This function loops through the list of images and labels and
        just calls the extract_embeddings fucntion to generate and store
        list_of_embeddings and list_of_metadata

        Args:
            images_path (_type_): _description_
            labels_path (_type_): _description_
        """

        label_files = [f for f in os.listdir(labels_path) if f.endswith(".json")]

        print(f"Found {len(label_files)} label files. Starting processing...")

        for label_file in tqdm(label_files):
            file_id = os.path.splitext(label_file)[0]
            full_label_path = os.path.join(labels_path, label_file)
            full_image_path = os.path.join(images_path, f"{file_id}.jpg")

            if os.path.exists(full_image_path):
                print(f"Processing pair: {file_id}")
                self.extract_embeddings(full_label_path, full_image_path)
            else:
                print(f"Warning: Image missing for label {label_file}. Skipping...")

        print("Batch processing complete.")

        print("clustering started")

        unique_categories = set(m["category"] for m in self.list_of_metadata)

        for category in unique_categories:
            print(f"Processing category: {category}")
            self.run_clustering(category)

        print("Saving results...")
        self.save_qc_results(images_path=images_path, output_dir=output_dir)

    def run_clustering(self, target_category):
        category_data = [
            (emb, m)
            for emb, m in zip(self.list_embeddings, self.list_of_metadata)
            if m["category"] == target_category
        ]

        if not category_data:
            print(f"No data for {target_category}")
            return

        category_embeddings, category_metadata = zip(*category_data)
        cat_emb_array = np.vstack(category_embeddings)
        category_embeddings_normalized = normalize(cat_emb_array, axis=1)

        n_samples = len(category_embeddings_normalized)

        if n_samples < 2:
            print(f"Skipping {target_category}: only {n_samples} sample(s).")
            for i in range(n_samples):
                category_metadata[i]["statistical_score"] = 0.0
            return

        calculated_k = int((n_samples / 2) ** 0.5)
        k = max(2, min(calculated_k, 30, n_samples))

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(category_embeddings_normalized)
        centroids = model.cluster_centers_

        for i in range(n_samples):
            embedding = category_embeddings_normalized[i]
            cluster_idx = cluster_labels[i]
            center = centroids[cluster_idx]

            distance = np.linalg.norm(embedding - center)
            category_metadata[i]["statistical_score"] = float(distance)

    def save_qc_results(
        self, images_path, output_dir="data/processed/k_means_qc", percentile=95
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        outlier_dir = os.path.join(run_dir, "outlier_crops")
        os.makedirs(outlier_dir, exist_ok=True)

        scores = [
            m["statistical_score"]
            for m in self.list_of_metadata
            if "statistical_score" in m
        ]
        if not scores:
            print("No scores found. Please run run_clustering first.")
            return
        threshold = np.percentile(scores, percentile)

        print(
            f"Flagging outliers at {percentile}th percentile (Score >= {threshold:.4f})"
        )
        for m in self.list_of_metadata:
            score = m.get("statistical_score", 0)
            if score >= threshold:
                m["is_outlier"] = True
                full_img_path = os.path.join(images_path, m["file"])
                with Image.open(full_img_path) as img:
                    box = m.get("box")
                    if box:
                        crop = img.crop(box)
                        crop_filename = (
                            f"{m['category']}_{m['obj_id']}_score_{score:.2f}.jpg"
                        )
                        crop.save(os.path.join(outlier_dir, crop_filename))
                        m["saved_crop_path"] = crop_filename
            else:
                m["is_outlier"] = False

        df = pd.DataFrame(self.list_of_metadata)
        report_path = os.path.join(run_dir, f"qc_report_{timestamp}.csv")
        df.to_csv(report_path, index=False)
        print(f"QC Report and crops saved to {run_dir}")
