import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model, EfficientNetB0Transfer
from ml.src.calibration import (
    TemperatureScaling,
    compute_nll,
    compute_brier_score,
    compute_ece,
    compute_accuracy_metrics,
    plot_reliability_diagram
)

CHECKPOINT_PATH = Path("artifacts/ml/efficientnet_b0_best.pth")
CALIBRATION_OUTPUT = Path("artifacts/ml/temperature_scaling.json")
SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
ARTIFACTS_DIR = Path("artifacts/ml")


def extract_logits_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    all_img_ids = []

    with torch.no_grad():
        for images, targets, img_ids in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(targets.numpy())
            all_img_ids.extend(img_ids)

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return logits_arr, labels_arr, all_img_ids


def main():
    print("=" * 75)
    print("DERMAI - EFFICIENTNET-B0 CONFIDENCE CALIBRATION & EVALUATION")
    print("Post-Hoc Temperature Scaling Optimization & Reliability Assessment")
    print("=" * 75)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Model
    print("\n[Step 1/5] Loading EfficientNet-B0 checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = get_model("efficientnet_b0", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint epoch: {ckpt.get('epoch')}, val balanced acc: {ckpt.get('val_balanced_acc'):.4f}")

    # 2. Extract Validation Split Logits
    print("\n[Step 2/5] Evaluating on Validation split (329 samples) to extract logits...")
    val_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="validation")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_logits, val_labels, _ = extract_logits_and_labels(model, val_loader, device)

    val_raw_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_raw_preds = np.argmax(val_raw_probs, axis=1)
    val_raw_nll = compute_nll(val_logits, val_labels)
    val_raw_brier = compute_brier_score(val_raw_probs, val_labels)
    val_raw_ece, _ = compute_ece(val_raw_probs, val_labels, n_bins=10)
    val_raw_acc, val_raw_bal_acc = compute_accuracy_metrics(val_raw_preds, val_labels)

    print(f"Validation Raw Metrics:")
    print(f"  Accuracy:         {val_raw_acc*100:.2f}%")
    print(f"  Balanced Acc:     {val_raw_bal_acc*100:.2f}%")
    print(f"  NLL (Loss):       {val_raw_nll:.4f}")
    print(f"  Brier Score:      {val_raw_brier:.4f}")
    print(f"  ECE (10 bins):    {val_raw_ece*100:.2f}%")

    # 3. Fit Temperature Scaling on Validation Split ONLY
    print("\n[Step 3/5] Fitting Temperature Scaling parameter T strictly on Validation logits...")
    calibrator = TemperatureScaling()
    fitted_temp = calibrator.fit(val_logits, val_labels)
    print(f"Optimal Fitted Temperature: T = {fitted_temp:.4f}")

    val_cal_probs = calibrator.predict_proba(torch.tensor(val_logits))
    val_cal_preds = np.argmax(val_cal_probs, axis=1)
    val_cal_nll = compute_nll(calibrator.calibrate_logits(torch.tensor(val_logits)).numpy(), val_labels)
    val_cal_brier = compute_brier_score(val_cal_probs, val_labels)
    val_cal_ece, _ = compute_ece(val_cal_probs, val_labels, n_bins=10)
    val_cal_acc, val_cal_bal_acc = compute_accuracy_metrics(val_cal_preds, val_labels)

    print(f"Validation Calibrated Metrics:")
    print(f"  Accuracy:         {val_cal_acc*100:.2f}% (rank preserved)")
    print(f"  Balanced Acc:     {val_cal_bal_acc*100:.2f}%")
    print(f"  NLL (Loss):       {val_cal_nll:.4f} (improvement: {val_raw_nll - val_cal_nll:+.4f})")
    print(f"  Brier Score:      {val_cal_brier:.4f} (improvement: {val_raw_brier - val_cal_brier:+.4f})")
    print(f"  ECE (10 bins):    {val_cal_ece*100:.2f}% (improvement: {(val_raw_ece - val_cal_ece)*100:+.2f}%)")

    # Save calibration parameter
    calibrator.save(CALIBRATION_OUTPUT, metadata={
        "model_architecture": "efficientnet_b0",
        "checkpoint": str(CHECKPOINT_PATH),
        "validation_samples": len(val_labels),
        "val_raw_ece": round(val_raw_ece, 4),
        "val_calibrated_ece": round(val_cal_ece, 4),
        "val_raw_nll": round(val_raw_nll, 4),
        "val_calibrated_nll": round(val_cal_nll, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })
    print(f"Saved calibration parameters to {CALIBRATION_OUTPUT}")

    # 4. Held-out Test Set Evaluation (Unchanged test set)
    print("\n[Step 4/5] Evaluating Raw vs Calibrated on Held-out Test Split (329 samples)...")
    test_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_logits, test_labels, test_img_ids = extract_logits_and_labels(model, test_loader, device)

    # Raw Test Metrics
    test_raw_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_raw_preds = np.argmax(test_raw_probs, axis=1)
    test_raw_nll = compute_nll(test_logits, test_labels)
    test_raw_brier = compute_brier_score(test_raw_probs, test_labels)
    test_raw_ece, test_raw_bins = compute_ece(test_raw_probs, test_labels, n_bins=10)
    test_raw_acc, test_raw_bal_acc = compute_accuracy_metrics(test_raw_preds, test_labels)

    # Calibrated Test Metrics
    test_cal_probs = calibrator.predict_proba(torch.tensor(test_logits))
    test_cal_preds = np.argmax(test_cal_probs, axis=1)
    test_cal_logits = calibrator.calibrate_logits(torch.tensor(test_logits)).numpy()
    test_cal_nll = compute_nll(test_cal_logits, test_labels)
    test_cal_brier = compute_brier_score(test_cal_probs, test_labels)
    test_cal_ece, test_cal_bins = compute_ece(test_cal_probs, test_labels, n_bins=10)
    test_cal_acc, test_cal_bal_acc = compute_accuracy_metrics(test_cal_preds, test_labels)

    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET EVALUATION SUMMARY (N = 329):")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Raw (Softmax)':<15} | {'Calibrated (T=' + f'{fitted_temp:.3f})':<15} | {'Delta':<12}")
    print("-" * 75)
    print(f"{'Negative Log-Likelihood':<25} | {test_raw_nll:<15.4f} | {test_cal_nll:<15.4f} | {test_cal_nll - test_raw_nll:<+12.4f}")
    print(f"{'Brier Score':<25} | {test_raw_brier:<15.4f} | {test_cal_brier:<15.4f} | {test_cal_brier - test_raw_brier:<+12.4f}")
    print(f"{'Expected Calib Error':<25} | {test_raw_ece*100:<14.2f}% | {test_cal_ece*100:<14.2f}% | {(test_cal_ece - test_raw_ece)*100:<+11.2f}%")
    print(f"{'Overall Accuracy':<25} | {test_raw_acc*100:<14.2f}% | {test_cal_acc*100:<14.2f}% | {(test_cal_acc - test_raw_acc)*100:<+11.2f}%")
    print(f"{'Balanced Accuracy':<25} | {test_raw_bal_acc*100:<14.2f}% | {test_cal_bal_acc*100:<14.2f}% | {(test_cal_bal_acc - test_raw_bal_acc)*100:<+11.2f}%")
    print("-" * 75)

    # 5. Generate Reliability Diagrams
    print("\n[Step 5/5] Generating Empirical Reliability Diagrams...")
    raw_plot_path = ARTIFACTS_DIR / "reliability_diagram_raw.png"
    cal_plot_path = ARTIFACTS_DIR / "reliability_diagram_calibrated.png"

    plot_reliability_diagram(
        probs=test_raw_probs,
        labels=test_labels,
        title="Reliability Diagram: EfficientNet-B0 (Uncalibrated Softmax)",
        output_path=raw_plot_path,
        n_bins=10,
        ece=test_raw_ece
    )
    print(f"Saved raw reliability diagram: {raw_plot_path}")

    plot_reliability_diagram(
        probs=test_cal_probs,
        labels=test_labels,
        title=f"Reliability Diagram: EfficientNet-B0 (Calibrated, T = {fitted_temp:.3f})",
        output_path=cal_plot_path,
        n_bins=10,
        ece=test_cal_ece
    )
    print(f"Saved calibrated reliability diagram: {cal_plot_path}")

    # Save complete test calibration report JSON
    report_json_path = ARTIFACTS_DIR / "confidence_calibration_results.json"
    full_results = {
        "model_architecture": "efficientnet_b0",
        "checkpoint": str(CHECKPOINT_PATH),
        "temperature": round(fitted_temp, 5),
        "validation_samples": len(val_labels),
        "test_samples": len(test_labels),
        "test_metrics": {
            "raw": {
                "nll": round(test_raw_nll, 4),
                "brier_score": round(test_raw_brier, 4),
                "ece": round(test_raw_ece, 4),
                "accuracy": round(test_raw_acc, 4),
                "balanced_accuracy": round(test_raw_bal_acc, 4),
                "bins": test_raw_bins
            },
            "calibrated": {
                "nll": round(test_cal_nll, 4),
                "brier_score": round(test_cal_brier, 4),
                "ece": round(test_cal_ece, 4),
                "accuracy": round(test_cal_acc, 4),
                "balanced_accuracy": round(test_cal_bal_acc, 4),
                "bins": test_cal_bins
            }
        },
        "artifacts": {
            "temperature_json": str(CALIBRATION_OUTPUT),
            "raw_reliability_plot": str(raw_plot_path),
            "calibrated_reliability_plot": str(cal_plot_path)
        }
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full calibration evaluation report to {report_json_path}")


if __name__ == "__main__":
    main()

