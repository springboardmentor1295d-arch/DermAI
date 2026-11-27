"""Create a PATIENT-LEVEL train/val/test split (prevents leakage between images of the same patient).
Usage: python -m ml.split --data-root /path/to/pad-ufes-20 --out ml/splits.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder containing metadata.csv and imgs_part_*/")
    ap.add_argument("--out", default="ml/splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.data_root)
    df = pd.read_csv(next(root.rglob("metadata.csv")))
    files = {p.name: str(p) for ext in ("*.png", "*.jpg", "*.jpeg") for p in root.rglob(ext)}
    df["path"] = df["img_id"].map(files)
    missing = df["path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} metadata rows have no image file; dropping them")
        df = df.dropna(subset=["path"])
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed)
    df["fold"] = -1
    for f, (_, idx) in enumerate(sgkf.split(df, df["diagnostic"], groups=df["patient_id"])):
        df.iloc[idx, df.columns.get_loc("fold")] = f
    df["split"] = df["fold"].map(lambda f: "test" if f == 0 else "val" if f == 1 else "train")
    df.to_csv(a.out, index=False)
    print(df["split"].value_counts())
    print(pd.crosstab(df["split"], df["diagnostic"]))
    assert not (set(df[df.split == "train"].patient_id) & set(df[df.split == "test"].patient_id)), "LEAKAGE"


if __name__ == "__main__":
    main()
