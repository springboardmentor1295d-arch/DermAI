import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.evaluate import evaluate_model


def main():
    full_split_csv = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
    images_dir = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
    ckpt_path = Path("artifacts/ml/best_model.pth")
    baseline_out = Path("outputs/mentor_submission/baseline_mobilenetv3")
    pred_samples_dir = baseline_out / "prediction_samples"
    baseline_out.mkdir(parents=True, exist_ok=True)
    pred_samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = get_model(architecture="mobilenetv3_large", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("Evaluating MobileNetV3-Large on test split (N=329)...")
    test_dataset = PadUfesDataset(split_csv_path=full_split_csv, images_dir=images_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    eval_res = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=baseline_out,
        split_name="test"
    )

    # Convert training history JSON to CSV
    hist_path = Path("artifacts/ml/mobilenetv3_large_history.json")
    with open(hist_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)
    pd.DataFrame(hist_data["history"]).to_csv(baseline_out / "training_history.csv", index=False)

    # Load test predictions and compute weighted F1
    preds_df = pd.read_csv(baseline_out / "test_predictions.csv")
    weighted_f1 = float(f1_score(preds_df["true_label"], preds_df["predicted_label"], average="weighted"))

    with open(baseline_out / "test_metrics.json", "r", encoding="utf-8") as f:
        test_metrics = json.load(f)
    test_metrics["weighted_f1"] = round(weighted_f1, 4)

    with open(baseline_out / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Standardize confusion matrix names
    if (baseline_out / "test_confusion_matrix.csv").exists():
        shutil.copyfile(baseline_out / "test_confusion_matrix.csv", baseline_out / "confusion_matrix.csv")
    if (baseline_out / "test_confusion_matrix.png").exists():
        shutil.copyfile(baseline_out / "test_confusion_matrix.png", baseline_out / "confusion_matrix.png")

    # Save 6 representative visual prediction samples (1 per class)
    img_lookup = {f.name.lower(): f for f in images_dir.rglob("*.png")}
    sample_preds_info = []
    for cls in DIAGNOSTIC_CLASSES:
        sample_row = preds_df[preds_df["true_label"] == cls].iloc[0]
        img_id = sample_row["img_id"]
        pred_label = sample_row["predicted_label"]
        orig_path = img_lookup[img_id.lower()]
        dest_filename = f"{cls}_pred_{pred_label}_{img_id}"
        dest_img = pred_samples_dir / dest_filename
        shutil.copyfile(orig_path, dest_img)
        sample_preds_info.append({
            "img_id": img_id,
            "true_class": cls,
            "true_class_name": CLASS_NAMES[cls],
            "predicted_class": pred_label,
            "predicted_class_name": CLASS_NAMES[pred_label],
            "confidence": float(sample_row["confidence"]),
            "is_correct": bool(cls == pred_label),
            "sample_file": dest_filename
        })

    with open(pred_samples_dir / "samples_index.json", "w", encoding="utf-8") as f:
        json.dump(sample_preds_info, f, indent=2)

    # Save config.json
    config = {
        "model_name": "MobileNetV3-Large Transfer Learning Classifier",
        "architecture": "mobilenetv3_large",
        "pretrained_weights": "ImageNet-1K (MobileNet_V3_Large_Weights.DEFAULT)",
        "input_size": [224, 224],
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "num_classes": 6,
        "classes": DIAGNOSTIC_CLASSES,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "lr_scheduler": "CosineAnnealingLR",
        "epochs": 3,
        "batch_size": 32,
        "loss_function": "Weighted Cross-Entropy (Balanced Inverse Frequency)",
        "random_seed": 42,
        "execution_device": str(device),
        "checkpoint_path": str(ckpt_path),
        "best_val_balanced_acc": hist_data["best_val_balanced_acc"],
        "test_overall_accuracy": test_metrics["overall_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_weighted_f1": test_metrics["weighted_f1"]
    }
    with open(baseline_out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Baseline evaluation completed successfully!")
    print(f"Overall Accuracy:  {test_metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1-Score:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
