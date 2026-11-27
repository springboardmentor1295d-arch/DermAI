"""Exploratory data analysis. Usage: python -m ml.eda --splits ml/splits.csv --out ml/eda"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="ml/splits.csv")
    ap.add_argument("--out", default="ml/eda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.splits)
    print(df.isna().mean().sort_values(ascending=False).head(15))
    for name, col in [("class_distribution", "diagnostic"), ("body_region", "region"), ("gender", "gender")]:
        df[col].value_counts().plot.bar(title=name); plt.tight_layout()
        plt.savefig(out / f"{name}.png"); plt.close()
    df["age"].plot.hist(bins=30, title="age"); plt.savefig(out / "age.png"); plt.close()
    pd.crosstab(df["split"], df["diagnostic"]).to_csv(out / "split_by_class.csv")
    classes = sorted(df["diagnostic"].unique())
    fig, ax = plt.subplots(len(classes), 4, figsize=(10, 2.5 * len(classes)))
    for i, c in enumerate(classes):
        for j, p in enumerate(df[df.diagnostic == c]["path"].head(4)):
            ax[i, j].imshow(Image.open(p)); ax[i, j].axis("off")
        ax[i, 0].set_title(c, loc="left")
    plt.tight_layout(); plt.savefig(out / "examples.png"); plt.close()
    sizes = [Image.open(p).size for p in df["path"].sample(min(200, len(df)), random_state=0)]
    print("Sample image sizes (w,h):", pd.Series(sizes).value_counts().head())
    print(f"EDA written to {out}")


if __name__ == "__main__":
    main()
