from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "Bone_Fracture_Binary_Classification"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_split(split_name: str):
    split_dir = DATASET_DIR / split_name

    normal_dir = split_dir / "not fractured"
    fracture_dir = split_dir / "fractured"

    rows = []

    print(f"Checking {split_name} split...")
    print(f"Normal folder: {normal_dir}")
    print(f"Fracture folder: {fracture_dir}")

    if not normal_dir.exists():
        print(f"WARNING: normal folder not found: {normal_dir}")

    if not fracture_dir.exists():
        print(f"WARNING: fracture folder not found: {fracture_dir}")

    for path in normal_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"image_path": str(path), "label": 0})

    for path in fracture_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"image_path": str(path), "label": 1})

    return pd.DataFrame(rows)


def main():
    print("Starting split creation...")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset folder: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        raise RuntimeError(f"Dataset folder not found: {DATASET_DIR}")

    for split_name in ["train", "val", "test"]:
        df = collect_split(split_name)

        if len(df) == 0:
            raise RuntimeError(f"No images found for split: {split_name}")

        output_path = SPLITS_DIR / f"{split_name}.csv"
        df.to_csv(output_path, index=False)

        print(f"\n{split_name.upper()} split")
        print("-" * 20)
        print(df["label"].value_counts())
        print(f"Total images: {len(df)}")
        print(f"Saved to: {output_path}")

    print("\nDone. CSV files created successfully.")


if __name__ == "__main__":
    main()
