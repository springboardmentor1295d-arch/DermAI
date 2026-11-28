import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.calibration import compute_nll, compute_brier_score, compute_ece

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
EFF_CKPT = Path("artifacts/ml/efficientnet_b0_best.pth")
MOB_CKPT = Path("artifacts/ml/mobilenetv3_large_best.pth")
OUTPUT_PATH = Path("artifacts/ml/ensemble_evaluation_results.json")


def main():
    print("=" * 75)
    print("DERMAI - ADVANCED AI AUDIT: ENSEMBLE EVALUATION")
    print("Evaluating Soft-Voting Ensemble: EfficientNet-B0 (0.6) + MobileNetV3 (0.4)")
    print("Evaluation Split: Held-Out Test Set (N=329)")
    print("=" * 75)

    device = torch.device("cpu")

    # Load test dataset
    test_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 1. Load EfficientNet-B0
    ckpt_eff = torch.load(EFF_CKPT, map_location=device)
    model_eff = get_model("efficientnet_b0", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model_eff.load_state_dict(ckpt_eff["model_state_dict"])
    model_eff.to(device)
    model_eff.eval()

    # 2. Load MobileNetV3-Large
    ckpt_mob = torch.load(MOB_CKPT, map_location=device)
    model_mob = get_model("mobilenetv3_large", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model_mob.load_state_dict(ckpt_mob["model_state_dict"])
    model_mob.to(device)
    model_mob.eval()

    # 3. Extract probabilities from both models
    eff_probs_list = []
    mob_probs_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            out_eff = model_eff(images)
            out_mob = model_mob(images)

            p_eff = torch.softmax(out_eff, dim=1).cpu().numpy()
            p_mob = torch.softmax(out_mob, dim=1).cpu().numpy()

            eff_probs_list.append(p_eff)
            mob_probs_list.append(p_mob)
            labels_list.append(targets.numpy())

    eff_probs = np.concatenate(eff_probs_list, axis=0)
    mob_probs = np.concatenate(mob_probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Standalone metrics
    eff_preds = np.argmax(eff_probs, axis=1)
    mob_preds = np.argmax(mob_probs, axis=1)

    eff_acc = float(np.mean(eff_preds == labels))
    eff_bal_acc = float(balanced_accuracy_score(labels, eff_preds))
    mob_acc = float(np.mean(mob_preds == labels))
    mob_bal_acc = float(balanced_accuracy_score(labels, mob_preds))

    # Ensemble: Weighted soft voting (0.6 EffNet + 0.4 MobileNet)
    ens_probs = 0.6 * eff_probs + 0.4 * mob_probs
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_acc = float(np.mean(ens_preds == labels))
    ens_bal_acc = float(balanced_accuracy_score(labels, ens_preds))
    ens_brier = compute_brier_score(ens_probs, labels)
    ens_ece, _ = compute_ece(ens_probs, labels, n_bins=10)

    print("\n" + "=" * 75)
    print("ENSEMBLE VS INDIVIDUAL MODELS ON HELD-OUT TEST SPLIT (N=329):")
    print("=" * 75)
    print(f"{'Model Architecture':<30} | {'Accuracy':<14} | {'Balanced Acc':<14}")
    print("-" * 75)
    print(f"{'MobileNetV3-Large':<30} | {mob_acc*100:<13.2f}% | {mob_bal_acc*100:<13.2f}%")
    print(f"{'EfficientNet-B0 (Deployed)':<30} | {eff_acc*100:<13.2f}% | {eff_bal_acc*100:<13.2f}%")
    print(f"{'Ensemble (0.6 Eff + 0.4 Mob)':<30} | {ens_acc*100:<13.2f}% | {ens_bal_acc*100:<13.2f}%")
    print("-" * 75)

    results = {
        "evaluation_split": "test",
        "sample_count": len(labels),
        "models": {
            "mobilenetv3_large": {
                "checkpoint": str(MOB_CKPT),
                "accuracy": round(mob_acc, 4),
                "balanced_accuracy": round(mob_bal_acc, 4)
            },
            "efficientnet_b0": {
                "checkpoint": str(EFF_CKPT),
                "accuracy": round(eff_acc, 4),
                "balanced_accuracy": round(eff_bal_acc, 4)
            },
            "ensemble_soft_voting": {
                "weights": {"efficientnet_b0": 0.6, "mobilenetv3_large": 0.4},
                "accuracy": round(ens_acc, 4),
                "balanced_accuracy": round(ens_bal_acc, 4),
                "brier_score": round(ens_brier, 4),
                "ece": round(ens_ece, 4)
            }
        },
        "critical_model_selection_policy": (
            "EfficientNet-B0 remains the sole deployed model in DermAI backend and frontend. "
            "Ensemble evaluated strictly as an offline research comparison."
        )
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ensemble evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

