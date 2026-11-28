from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, CLASS_NAMES


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Optional[Path] = None,
    split_name: str = "test"
) -> Dict[str, Any]:
    """
    Evaluates a trained PyTorch model on a split without tuning.
    Generates genuine, reproducible metrics and confusion matrix artifacts.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_img_ids = []

    with torch.no_grad():
        for images, targets, img_ids in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            all_img_ids.extend(img_ids)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Core Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=list(range(len(DIAGNOSTIC_CLASSES))), zero_division=0
    )

    # 6x6 Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(DIAGNOSTIC_CLASSES))))

    per_class_summary = {}
    for i, cls in enumerate(DIAGNOSTIC_CLASSES):
        # Calculate one-vs-rest sensitivity and specificity
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class_summary[cls] = {
            "name": CLASS_NAMES[cls],
            "precision": round(float(class_p[i]), 4),
            "recall": round(float(class_r[i]), 4),
            "f1_score": round(float(class_f1[i]), 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "support": int(class_supp[i]),
        }

    results = {
        "split": split_name,
        "sample_count": len(all_targets),
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "classes": DIAGNOSTIC_CLASSES
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save JSON metrics
        with open(save_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Save Confusion Matrix CSV
        cm_df = pd.DataFrame(cm, index=DIAGNOSTIC_CLASSES, columns=DIAGNOSTIC_CLASSES)
        cm_df.to_csv(save_dir / f"{split_name}_confusion_matrix.csv")

        # 3. Save Per-Class Metrics CSV
        records = []
        for cls, m in per_class_summary.items():
            records.append({
                "code": cls,
                "name": m["name"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "support": m["support"]
            })
        pd.DataFrame(records).to_csv(save_dir / f"{split_name}_per_class_metrics.csv", index=False)

        # 4. Save individual prediction records CSV
        pred_records = []
        for idx in range(len(all_targets)):
            pred_records.append({
                "img_id": all_img_ids[idx],
                "true_label": IDX_TO_CLASS[all_targets[idx]],
                "predicted_label": IDX_TO_CLASS[all_preds[idx]],
                "confidence": round(float(all_probs[idx][all_preds[idx]]), 4),
                **{f"prob_{cls}": round(float(all_probs[idx][i]), 4) for i, cls in enumerate(DIAGNOSTIC_CLASSES)}
            })
        pd.DataFrame(pred_records).to_csv(save_dir / f"{split_name}_predictions.csv", index=False)

        # 5. Plot and save Confusion Matrix Figure
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=DIAGNOSTIC_CLASSES,
            yticklabels=DIAGNOSTIC_CLASSES,
            title=f"DermAI Confusion Matrix ({split_name.capitalize()} Split, N={len(all_targets)})",
            ylabel="True Diagnostic Class",
            xlabel="Predicted Diagnostic Class"
        )
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(save_dir / f"{split_name}_confusion_matrix.png", dpi=200)
        plt.close()

    return results
