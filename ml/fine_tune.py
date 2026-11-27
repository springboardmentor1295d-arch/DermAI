import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.dataset import CLASSES, PadDataset, train_tf, eval_tf
from ml.models import build_model
from ml.train import FocalLoss, predict, softmax
from ml.evaluate import (
    compute_metrics,
    fit_temperature,
    plot_confusion
)


def main():

    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = Path(
        "ml/runs/efficientnet_b0_fusion/efficientnet_b0_meta"
    )

    out = Path(
        "ml/runs/efficientnet_b0_finetuned/efficientnet_b0_meta"
    )

    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        source / "best.pt",
        map_location="cpu",
        weights_only=True
    )

    encoder = joblib.load(
        source / "meta_encoder.joblib"
    )

    assert list(checkpoint["classes"]) == CLASSES
    assert checkpoint["use_meta"]

    df = pd.read_csv("ml/splits.csv")

    size = checkpoint["img_size"]

    train_data = PadDataset(
        df[df["split"] == "train"],
        train_tf(size),
        encoder
    )

    val_data = PadDataset(
        df[df["split"] == "val"],
        eval_tf(size),
        encoder
    )

    counts = np.bincount(
        train_data.labels,
        minlength=len(CLASSES)
    ).clip(min=1)

    weights = (1.0 / counts)[train_data.labels]

    sampler = WeightedRandomSampler(
        weights,
        len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        sampler=sampler,
        num_workers=2
    )

    val_loader = DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    model = build_model(
        checkpoint["arch"],
        len(CLASSES),
        pretrained=False,
        meta_dim=train_data.meta.shape[1]
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    criterion = FocalLoss(2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00001,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    logits, labels = predict(
        model, val_loader, device, True
    )

    baseline = compute_metrics(
        labels,
        softmax(logits),
        CLASSES
    )

    print("Original validation accuracy:",
          baseline["accuracy"])

    print("Original validation Macro F1:",
          baseline["macro_f1"])

    best = baseline
    bad = 0
    history = []
    improved = False

    torch.save(
        model.state_dict(),
        out / "_best_state.pt"
    )

    print("\nStarting fine-tuning...")

    for epoch in range(10):

        start = time.time()
        model.train()

        for module in model.modules():
            if isinstance(
                module,
                torch.nn.modules.batchnorm._BatchNorm
            ):
                module.eval()

        total_loss = 0

        for x, m, y in train_loader:

            x = x.to(device)
            m = m.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x, m)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)

        scheduler.step()

        logits, labels = predict(
            model, val_loader, device, True
        )

        metrics = compute_metrics(
            labels,
            softmax(logits),
            CLASSES
        )

        record = {
            "epoch": epoch + 1,
            "loss": total_loss / len(train_data),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_bal_acc": metrics["balanced_accuracy"],
            "sec": round(time.time() - start, 1)
        }

        history.append(record)
        print(record, flush=True)

        acceptable = (
            metrics["macro_f1"]
            >= baseline["macro_f1"]
            and metrics["balanced_accuracy"]
            >= baseline["balanced_accuracy"]
            and metrics["per_class"]["MEL"]["sensitivity"]
            >= baseline["per_class"]["MEL"]["sensitivity"]
            and metrics["per_class"]["SCC"]["sensitivity"]
            >= baseline["per_class"]["SCC"]["sensitivity"]
        )

        if (
            metrics["accuracy"] > best["accuracy"]
            and acceptable
        ):

            best = metrics
            bad = 0
            improved = True

            torch.save(
                model.state_dict(),
                out / "_best_state.pt"
            )

            print("Better model saved")

        else:

            bad += 1

            if bad >= 4:

                print("Early stopping")
                break

    model.load_state_dict(
        torch.load(
            out / "_best_state.pt",
            map_location=device,
            weights_only=True
        )
    )

    logits, labels = predict(
        model, val_loader, device, True
    )

    temperature = fit_temperature(
        logits, labels
    )

    probs = softmax(logits / temperature)

    metrics = compute_metrics(
        labels, probs, CLASSES
    )

    metrics["history"] = history
    metrics["temperature"] = temperature
    metrics["split"] = "val"
    metrics["improved"] = improved

    with open(
        out / "validation_metrics.json", "w"
    ) as f:

        json.dump(metrics, f, indent=2)

    joblib.dump(
        encoder,
        out / "meta_encoder.joblib"
    )

    torch.save({

        "arch": checkpoint["arch"],
        "classes": CLASSES,
        "img_size": size,
        "use_meta": True,
        "temperature": temperature,

        "state_dict": {
            k: v.cpu()
            for k, v in model.state_dict().items()
        },

        "validation_metrics": {
            k: metrics[k]
            for k in [
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "mcc",
                "macro_auc_ovr"
            ]
        }

    }, out / "best.pt")

    plot_confusion(
        metrics["confusion_matrix"],
        CLASSES,
        out / "validation_confusion_matrix.png",
        "Fine-tuned EfficientNet-B0 Validation"
    )

    (out / "_best_state.pt").unlink(
        missing_ok=True
    )

    print("\nFINE-TUNING RESULTS")

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

    if improved:
        print("Fine-tuning improved validation accuracy.")
    else:
        print("No acceptable improvement. Original weights retained.")


if __name__ == "__main__":
    main()