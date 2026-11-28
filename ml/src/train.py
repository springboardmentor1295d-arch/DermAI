from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES
from ml.src.models import get_model
from ml.src.loss import get_loss_function


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, targets, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = float(np.mean(np.array(all_preds) == np.array(all_targets))) if total > 0 else 0.0
    val_bal_acc = float(balanced_accuracy_score(all_targets, all_preds)) if total > 0 else 0.0
    return val_loss, val_acc, val_bal_acc


def run_training_experiment(
    architecture: str = "mobilenetv3_large",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    split_csv_path: Path = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv"),
    images_dir: Path = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images"),
    output_dir: Path = Path("./artifacts/ml"),
    max_batches_per_epoch: Optional[int] = None,
    pilot_mode: bool = False
) -> Dict[str, Any]:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device} (Architecture: {architecture})")

    # Load datasets
    train_dataset = PadUfesDataset(split_csv_path=split_csv_path, images_dir=images_dir, split="train")
    val_dataset = PadUfesDataset(split_csv_path=split_csv_path, images_dir=images_dir, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, loss and optimizer
    model = get_model(architecture=architecture, num_classes=len(DIAGNOSTIC_CLASSES), pretrained=True).to(device)
    criterion = get_loss_function(split_csv_path=split_csv_path, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_bal_acc = -1.0
    best_model_path = output_dir / f"{architecture}_best.pth"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=max_batches_per_epoch
        )
        val_loss, val_acc, val_bal_acc = validate_epoch(
            model, val_loader, criterion, device, max_batches=max_batches_per_epoch
        )
        scheduler.step()
        epoch_dur = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_balanced_acc": round(val_bal_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "duration_sec": round(epoch_dur, 2)
        }
        history.append(record)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val BalAcc: {val_bal_acc:.4f} ({epoch_dur:.1f}s)")

        # Save checkpoint if best validation balanced accuracy
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save({
                "epoch": epoch,
                "architecture": architecture,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_balanced_acc": val_bal_acc,
                "classes": DIAGNOSTIC_CLASSES,
                "seed": seed
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    total_time = time.time() - start_time

    # Save training history JSON
    history_file = output_dir / f"{architecture}_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": architecture,
            "epochs": epochs,
            "total_time_sec": round(total_time, 2),
            "best_val_balanced_acc": round(best_val_bal_acc, 4),
            "history": history
        }, f, indent=2)

    return {
        "architecture": architecture,
        "best_val_balanced_acc": best_val_bal_acc,
        "best_model_path": str(best_model_path),
        "history": history
    }
