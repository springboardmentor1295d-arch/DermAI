import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ml.src.train import run_training_experiment, set_seed
from ml.src.dataset import DIAGNOSTIC_CLASSES

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
ARTIFACT_DIR = Path("./artifacts/ml")


def plot_loss_curve(history: list, output_path: Path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_bal_acc = [h["val_balanced_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color=color, fontsize=11)
    l1 = ax1.plot(epochs, train_loss, "o-", color="tab:red", label="Train Loss", linewidth=2)
    l2 = ax1.plot(epochs, val_loss, "s--", color="tab:orange", label="Val Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Balanced Accuracy", color=color, fontsize=11)
    l3 = ax2.plot(epochs, val_bal_acc, "^-", color="tab:blue", label="Val Balanced Acc", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    plt.title("EfficientNet-B0: Training & Validation Progression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved loss curve plot to {output_path}")


def main():
    print("=" * 70)
    print("DERMAI - EFFICIENTNET-B0 TRAINING PIPELINE")
    print("Pretrained EfficientNet-B0 Transfer Learning (ImageNet-1K)")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    epochs = 3
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    seed = 42

    start_time = time.time()
    train_res = run_training_experiment(
        architecture="efficientnet_b0",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        split_csv_path=SPLIT_CSV,
        images_dir=IMAGES_DIR,
        output_dir=ARTIFACT_DIR
    )
    total_duration = time.time() - start_time
    print(f"\nEfficientNet-B0 Training Complete in {total_duration:.1f}s.")
    print(f"Best Validation Balanced Accuracy: {train_res['best_val_balanced_acc']:.4f}")

    # Plot loss curves
    history = train_res["history"]
    plot_loss_curve(history, ARTIFACT_DIR / "efficientnet_b0_loss_curve.png")

    # Save detailed training configuration
    config = {
        "model_name": "EfficientNet-B0 Transfer Learning Classifier",
        "architecture": "efficientnet_b0",
        "pretrained_weights": "ImageNet-1K (EfficientNet_B0_Weights.DEFAULT)",
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "num_classes": len(DIAGNOSTIC_CLASSES),
        "classes": DIAGNOSTIC_CLASSES,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "lr_scheduler": "CosineAnnealingLR",
        "epochs": epochs,
        "batch_size": batch_size,
        "loss_function": "Weighted Cross-Entropy (Balanced Inverse Frequency)",
        "random_seed": seed,
        "execution_device": str(device),
        "total_duration_sec": round(total_duration, 2),
        "best_val_balanced_acc": round(train_res["best_val_balanced_acc"], 4),
        "checkpoint_path": str(ARTIFACT_DIR / "efficientnet_b0_best.pth")
    }
    with open(ARTIFACT_DIR / "efficientnet_b0_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {ARTIFACT_DIR / 'efficientnet_b0_config.json'}")


if __name__ == "__main__":
    main()
