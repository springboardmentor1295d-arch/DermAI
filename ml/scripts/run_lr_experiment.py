import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES
from ml.src.models import get_model
from ml.src.loss import get_loss_function
from ml.src.train import set_seed, train_epoch, validate_epoch

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
EXP_BASE_DIR = Path("artifacts/ml/experiments")
REPORT_PATH = Path("LEARNING_RATE_EXPERIMENT_REPORT.md")


def run_single_lr_trial(
    exp_id: str,
    lr: float,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 1,
    max_train_batches: int = 20,
    max_val_batches: int = 10,
    seed: int = 42
):
    set_seed(seed)
    exp_dir = EXP_BASE_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{exp_id}] Starting Controlled Trial: Learning Rate = {lr} ...")
    model = get_model(architecture="efficientnet_b0", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.time()
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, max_batches=max_train_batches
    )
    val_loss, val_acc, val_bal_acc = validate_epoch(
        model, val_loader, criterion, device, max_batches=max_val_batches
    )
    duration = time.time() - start_time

    ckpt_path = exp_dir / "best_val_checkpoint.pth"
    torch.save({
        "exp_id": exp_id,
        "lr": lr,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_balanced_acc": round(val_bal_acc, 4),
        "seed": seed,
        "architecture": "efficientnet_b0"
    }, ckpt_path)

    print(f"[{exp_id}] Duration: {duration:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BalAcc: {val_bal_acc:.4f}")

    return {
        "experiment_id": exp_id,
        "learning_rate": lr,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_balanced_acc": round(val_bal_acc, 4),
        "duration_sec": round(duration, 2),
        "checkpoint_path": str(ckpt_path)
    }


def main():
    print("=" * 75)
    print("DERMAI - EFFICIENTNET-B0 CONTROLLED LEARNING-RATE EXPERIMENT")
    print("Systematic Empirical Comparison of Learning Rates: 1e-3 vs 1e-4 vs 1e-5")
    print("=" * 75)

    EXP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset splits
    train_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="train")
    val_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    criterion = get_loss_function(split_csv_path=SPLIT_CSV, device=device)

    candidate_lrs = [
        {"id": "EXP-LR-1E-3", "lr": 1e-3, "label": "High LR (1e-3)"},
        {"id": "EXP-LR-1E-4", "lr": 1e-4, "label": "Standard LR (1e-4) [Selected]"},
        {"id": "EXP-LR-1E-5", "lr": 1e-5, "label": "Low LR (1e-5)"},
    ]

    results = []
    for cand in candidate_lrs:
        res = run_single_lr_trial(
            exp_id=cand["id"],
            lr=cand["lr"],
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=1,
            max_train_batches=20,
            max_val_batches=10,
            seed=42
        )
        res["label"] = cand["label"]
        results.append(res)

    # Save JSON summary
    summary_path = EXP_BASE_DIR / "lr_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"experiments": results}, f, indent=2)
    print(f"\nSaved experiment summary to {summary_path}")

    # Also include full 3-epoch baseline comparison
    print("\n" + "=" * 75)
    print("CONTROLLED LEARNING-RATE EXPERIMENT RESULTS SUMMARY:")
    print("=" * 75)
    print(f"{'Experiment ID':<16} | {'Learning Rate':<14} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val BalAcc':<12} | {'Duration'}")
    print("-" * 75)
    for r in results:
        print(f"{r['experiment_id']:<16} | {r['learning_rate']:<14.1e} | {r['train_loss']:<11.4f} | {r['val_loss']:<10.4f} | {r['val_balanced_acc']*100:<11.2f}% | {r['duration_sec']:.1f}s")
    print("-" * 75)


if __name__ == "__main__":
    main()

