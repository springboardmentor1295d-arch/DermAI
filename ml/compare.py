"""Train several architectures, tabulate results, and evaluate a softmax-average ensemble.
python -m ml.compare --archs resnet50 efficientnet_b0 densenet121 convnext_tiny swin_tiny_patch4_window7_224
python -m ml.compare --ensemble-only --archs efficientnet_b0 convnext_tiny
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import CLASSES
from .evaluate import compute_metrics, plot_confusion
from .models import build_model
from .train import make_loaders, predict, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["resnet50", "densenet121", "efficientnet_b0",
                                                   "mobilenetv3_large_100", "convnext_tiny",
                                                   "swin_tiny_patch4_window7_224"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--splits", default="ml/splits.csv")
    ap.add_argument("--out", default="ml/runs")
    ap.add_argument("--ensemble-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.ensemble_only:
        for arch in a.archs:
            subprocess.run([sys.executable, "-m", "ml.train", "--arch", arch, "--epochs", str(a.epochs),
                            "--splits", a.splits, "--out", a.out, "--mlflow"], check=True)
    rows = []
    for arch in a.archs:
        m = json.load(open(out / arch / "metrics.json"))
        rows.append(dict(model=arch, macro_f1=m["macro_f1"], balanced_acc=m["balanced_accuracy"], mcc=m["mcc"],
                         auc=m["macro_auc_ovr"], accuracy=m["accuracy"], ece=m.get("ece"),
                         mel_recall=m["per_class"]["MEL"]["sensitivity"]))
    # ensemble
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.splits)
    _, _, l_te, _, _ = make_loaders(df, 224, 32, 2, False, False)
    P = []
    for arch in a.archs:
        ck = torch.load(out / arch / "best.pt", map_location=dev)
        mdl = build_model(arch, len(CLASSES), False).to(dev); mdl.load_state_dict(ck["state_dict"])
        lg, y = predict(mdl, l_te, dev, False); P.append(softmax(lg / ck["temperature"]))
    ens = compute_metrics(y, np.mean(P, 0), CLASSES)
    rows.append(dict(model="ENSEMBLE(" + "+".join(a.archs) + ")", macro_f1=ens["macro_f1"],
                     balanced_acc=ens["balanced_accuracy"], mcc=ens["mcc"], auc=ens["macro_auc_ovr"],
                     accuracy=ens["accuracy"], ece=None, mel_recall=ens["per_class"]["MEL"]["sensitivity"]))
    plot_confusion(ens["confusion_matrix"], CLASSES, out / "ensemble_confusion.png", "Ensemble")
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(out / "comparison.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
