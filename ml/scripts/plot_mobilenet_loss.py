import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    history_file = Path("artifacts/ml/mobilenetv3_large_history.json")
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data["history"]
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

    plt.title("MobileNetV3-Large: Training & Validation Progression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = Path("artifacts/ml/mobilenetv3_large_loss_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved MobileNetV3 loss curve to {out_path}")

if __name__ == "__main__":
    main()
