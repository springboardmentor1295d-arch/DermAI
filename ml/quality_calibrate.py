"""Calibrate image-quality thresholds on YOUR data instead of guessing.
python -m ml.quality_calibrate --splits ml/splits.csv --n 400
Prints the distribution of blur/brightness/contrast and suggested thresholds for backend/app/services/quality.py."""
import argparse
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backend.app.services import quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="ml/splits.csv"); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    df = pd.read_csv(a.splits).sample(n=min(a.n, 10**9), random_state=0, replace=False) if a.n < len(pd.read_csv(a.splits)) else pd.read_csv(a.splits)
    rows = []
    for p in df["path"]:
        im = cv2.imread(p)
        if im is None:
            continue
        rows.append(quality.assess(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    q = pd.DataFrame(rows)
    print(q[["blur_score", "brightness", "contrast"]].describe(percentiles=[.02, .05, .1, .5, .9, .98]).round(1))
    print("\nLabels with CURRENT thresholds:\n", q["label"].value_counts(normalize=True).round(3))
    print("\nSuggested (reject roughly the worst 2% as Poor, next 8% as Acceptable):")
    print(f"  blur_poor={np.percentile(q.blur_score, 2):.0f}, blur_ok={np.percentile(q.blur_score, 10):.0f}")
    print(f"  dark_poor={np.percentile(q.brightness, 1):.0f}, bright_poor={np.percentile(q.brightness, 99):.0f}")
    print(f"  contrast_poor={np.percentile(q.contrast, 2):.0f}, contrast_ok={np.percentile(q.contrast, 10):.0f}")
    print("Paste these into the T dict in backend/app/services/quality.py")


if __name__ == "__main__":
    main()
