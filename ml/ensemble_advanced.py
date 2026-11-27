from pathlib import Path
import argparse
import json
import joblib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ml.dataset import PadDataset, eval_tf, CLASSES
from ml.models import build_model
from ml.evaluate import compute_metrics, plot_confusion

torch.set_num_threads(6)

paths = [
    Path("ml/runs/efficientnet_b0_fusion/efficientnet_b0_meta"),
    Path("ml/runs/efficientnet_b0_advanced/efficientnet_b0_meta")
]

df = pd.read_csv("ml/splits.csv")

@torch.inference_mode()
def predict(path, split):
    ckpt = torch.load(
        path / "best.pt",
        map_location="cpu",
        weights_only=True
    )

    encoder = joblib.load(path / "meta_encoder.joblib")

    dataset = PadDataset(
        df[df["split"] == split],
        eval_tf(ckpt["img_size"]),
        encoder
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    assert list(ckpt["classes"]) == CLASSES

    model = build_model(
        ckpt["arch"],
        len(CLASSES),
        pretrained=False,
        meta_dim=dataset.meta.shape[1]
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    original, tta2, tta4, labels = [], [], [], []
    temperature = ckpt.get("temperature", 1.0)

    for x, m, y in loader:

        p1 = torch.softmax(
            model(x, m) / temperature, dim=1
        )

        p2 = torch.softmax(
            model(torch.flip(x, [3]), m) / temperature,
            dim=1
        )

        p3 = torch.softmax(
            model(torch.flip(x, [2]), m) / temperature,
            dim=1
        )

        p4 = torch.softmax(
            model(torch.flip(x, [2, 3]), m) / temperature,
            dim=1
        )

        original.append(p1.numpy())
        tta2.append(((p1 + p2) / 2).numpy())
        tta4.append(((p1 + p2 + p3 + p4) / 4).numpy())
        labels.append(y.numpy())

    return {
        "original": np.concatenate(original),
        "tta2": np.concatenate(tta2),
        "tta4": np.concatenate(tta4)
    }, np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    print("Evaluating validation dataset...")

    v1, yv = predict(paths[0], "val")
    v2, yv2 = predict(paths[1], "val")

    assert np.array_equal(yv, yv2)

    baseline_probs = 0.6 * v1["tta2"] + 0.4 * v2["tta2"]
    baseline = compute_metrics(yv, baseline_probs, CLASSES)

    best = baseline
    best_weight = 0.6
    best_tta = "tta2"

    print("Previous validation accuracy:",
          baseline["accuracy"])

    print("Optimizing ensemble weights...")

    for mode in ["original", "tta2", "tta4"]:

        for w in np.linspace(0, 1, 21):

            probs = w * v1[mode] + (1 - w) * v2[mode]
            met = compute_metrics(yv, probs, CLASSES)

            if (
                met["accuracy"] > best["accuracy"]
                and met["macro_f1"] >= baseline["macro_f1"]
                and met["balanced_accuracy"] >= baseline["balanced_accuracy"]
                and met["per_class"]["MEL"]["sensitivity"] >= baseline["per_class"]["MEL"]["sensitivity"]
                and met["per_class"]["SCC"]["sensitivity"] >= baseline["per_class"]["SCC"]["sensitivity"]
            ):
                best = met
                best_weight = float(w)
                best_tta = mode

    print("\nBEST CONFIGURATION")
    print("Fusion weight:", best_weight)
    print("Advanced weight:", 1 - best_weight)
    print("TTA method:", best_tta)
    print("Validation accuracy:", best["accuracy"])
    print("Validation Macro F1:", best["macro_f1"])

    out = Path("ml/runs/efficientnet_b0_ensemble_advanced")
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "fusion_weight": best_weight,
        "advanced_weight": 1 - best_weight,
        "tta": best_tta,
        "validation_metrics": best
    }

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if best["accuracy"] <= baseline["accuracy"]:
        print("No improvement over the existing ensemble.")
        print("Keep your existing ensemble model.")
        return

    if not args.evaluate_test:
        print("Validation evaluation completed.")
        return

    print("\nEvaluating test dataset...")

    p1, y = predict(paths[0], "test")
    p2, y2 = predict(paths[1], "test")

    assert np.array_equal(y, y2)

    probs = (
        best_weight * p1[best_tta]
        + (1 - best_weight) * p2[best_tta]
    )

    metrics = compute_metrics(y, probs, CLASSES)

    metrics["fusion_weight"] = best_weight
    metrics["advanced_weight"] = 1 - best_weight
    metrics["tta"] = best_tta

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(
        metrics["confusion_matrix"],
        CLASSES,
        out / "confusion_matrix.png",
        "Advanced EfficientNet-B0 Ensemble"
    )

    print("\nFINAL ENSEMBLE RESULTS")

    for key in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "macro_auc_ovr"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("MEL recall:",
          metrics["per_class"]["MEL"]["sensitivity"])

    print("SCC recall:",
          metrics["per_class"]["SCC"]["sensitivity"])


if __name__ == "__main__":
    main()