import os

from PIL import Image

RUN_NAME = "traffic_vlm_v4_run_1"
BASE_DIR = os.path.join("outputs", RUN_NAME, "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_ORDER = [
    "attention_sample_0.png",
    "attention_sample_1.png",
    "attention_sample_2.png",
    "attention_sample_3.png",
    "failures.png",
]


def create_pdf_for_epoch(epoch_path):
    images = []

    for img_name in IMAGE_ORDER:
        img_path = os.path.join(epoch_path, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing {img_path}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        images.append(img)

    if not images:
        print(f"[SKIP] No images found in {epoch_path}")
        return

    epoch_name = os.path.basename(epoch_path)
    output_pdf = os.path.join(RESULTS_DIR, f"{epoch_name}.pdf")

    images[0].save(
        output_pdf,
        save_all=True,
        append_images=images[1:],
    )

    print(f"[OK] Created {output_pdf}")


def main():
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for folder in sorted(os.listdir(BASE_DIR)):
        if folder.startswith("epoch_"):
            epoch_path = os.path.join(BASE_DIR, folder)
            if os.path.isdir(epoch_path):
                create_pdf_for_epoch(epoch_path)


if __name__ == "__main__":
    main()
