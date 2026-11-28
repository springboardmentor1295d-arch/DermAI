import sys
from pathlib import Path
import pandas as pd
from PIL import Image

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")


def verify_split():
    print("=" * 60)
    print("VERIFYING PAD-UFES-20 PATIENT-LEVEL STRATIFIED SPLIT")
    print("=" * 60)

    if not SPLIT_CSV.exists():
        print(f"ERROR: Split CSV not found at {SPLIT_CSV}")
        sys.exit(1)

    df = pd.read_csv(SPLIT_CSV)
    print(f"Total records in split CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    required_cols = {"patient_id", "diagnostic", "img_id", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # 1. Check patient overlap
    patient_sets = {name: set(group["patient_id"]) for name, group in df.groupby("split")}
    splits = sorted(patient_sets.keys())
    leakage = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = patient_sets[splits[i]] & patient_sets[splits[j]]
            print(f"Patient overlap between {splits[i]} and {splits[j]}: {len(overlap)}")
            if len(overlap) > 0:
                leakage = True

    if leakage:
        print("ERROR: Patient leakage detected!")
        sys.exit(1)
    else:
        print("PASS: Zero patient overlap between splits.")

    # 2. Check duplicate image IDs
    dup_imgs = int(df["img_id"].duplicated().sum())
    print(f"Duplicate img_ids: {dup_imgs}")
    if dup_imgs > 0:
        print("ERROR: Duplicate image IDs found!")
        sys.exit(1)
    else:
        print("PASS: No duplicate image IDs.")

    # 3. Check split distribution
    counts = df.groupby(["split", "diagnostic"]).size().unstack(fill_value=0)
    print("\nClass distribution per split:")
    print(counts.to_string())

    # 4. Check image file existence
    print("\nScanning images directory recursively...")
    found_images = {}
    for f in IMAGES_DIR.rglob("*.png"):
        if f.is_file():
            found_images[f.name.lower()] = f

    print(f"Total image files discovered: {len(found_images)}")
    missing_files = []
    for img_id in df["img_id"]:
        if str(img_id).strip().lower() not in found_images:
            missing_files.append(img_id)

    if missing_files:
        print(f"ERROR: {len(missing_files)} image files not found on disk!")
        sys.exit(1)
    else:
        print("PASS: All 2,298 image files exist and match split manifest.")

    print("\nALL SPLIT INTEGRITY CHECKS PASSED.")


if __name__ == "__main__":
    verify_split()
