from pathlib import Path
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

    probs, augmented, labels = [], [], []
    temperature = ckpt.get("temperature", 1.0)

    for x, m, y in loader:
        p1 = torch.softmax(
            model(x, m) / temperature, dim=1
        )

        p2 = torch.softmax(
            model(torch.flip(x, [3]), m) / temperature,
            dim=1
        )

        probs.append(p1.numpy())
        augmented.append(((p1 + p2) / 2).numpy())
        labels.append(y.numpy())

    return (
        np.concatenate(probs),
        np.concatenate(augmented),
        np.concatenate(labels)
    )

print("Evaluating validation dataset...")

v1, t1, yv = predict(paths[0], "val")
v2, t2, yv2 = predict(paths[1], "val")

assert np.array_equal(yv, yv2)

baseline = compute_metrics(yv, v1, CLASSES)
best = baseline
best_weight = 1.0
best_tta = False

for tta in [False, True]:
    p1, p2 = (t1, t2) if tta else (v1, v2)

    for w in np.linspace(0, 1, 11):
        probs = w * p1 + (1 - w) * p2
        met = compute_metrics(yv, probs, CLASSES)

        if (
            met["accuracy"] > best["accuracy"]
            and met["macro_f1"] >= baseline["macro_f1"]
            and met["balanced_accuracy"] >=
                baseline["balanced_accuracy"]
        ):
            best = met
            best_weight = float(w)
            best_tta = tta

print("Selected weight:", best_weight)
print("Test-time augmentation:", best_tta)
print("Validation accuracy:", best["accuracy"])

print("Evaluating test dataset...")

p1, a1, y = predict(paths[0], "test")
p2, a2, y2 = predict(paths[1], "test")

assert np.array_equal(y, y2)

if best_tta:
    p1, p2 = a1, a2

probs = best_weight * p1 + (1 - best_weight) * p2

metrics = compute_metrics(y, probs, CLASSES)

metrics["fusion_weight"] = best_weight
metrics["advanced_weight"] = 1 - best_weight
metrics["tta"] = best_tta

out = Path("ml/runs/efficientnet_b0_ensemble")
out.mkdir(parents=True, exist_ok=True)

with open(out / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

plot_confusion(
    metrics["confusion_matrix"],
    CLASSES,
    out / "confusion_matrix.png",
    "EfficientNet-B0 Ensemble"
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

print("MEL recall:", metrics["per_class"]["MEL"]["sensitivity"])
print("SCC recall:", metrics["per_class"]["SCC"]["sensitivity"])