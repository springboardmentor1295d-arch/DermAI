from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import shutil
import time
import torch
from torch.utils.data import DataLoader

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.train import run_training_experiment, set_seed
from ml.src.models import get_model
from ml.src.evaluate import evaluate_model

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
ARTIFACT_DIR = Path("./artifacts/ml")


def main():
    print("=" * 70)
    print("DERMAI - MACHINE LEARNING EXPERIMENTS PIPELINE")
    print("=" * 70)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution compute device: {device}")

    # 1. Baseline Experiment (Baseline CNN)
    print("\n--- EXPERIMENT 1: BASELINE CNN ---")
    baseline_res = run_training_experiment(
        architecture="baseline",
        epochs=3,
        batch_size=32,
        lr=3e-4,
        seed=42,
        output_dir=ARTIFACT_DIR,
        split_csv_path=SPLIT_CSV,
        images_dir=IMAGES_DIR
    )
    print(f"Baseline CNN Training finished. Best Val Balanced Acc: {baseline_res['best_val_balanced_acc']:.4f}")

    # 2. Transfer Learning Experiment (MobileNetV3-Large)
    print("\n--- EXPERIMENT 2: TRANSFER LEARNING (MobileNetV3-Large) ---")
    transfer_res = run_training_experiment(
        architecture="mobilenetv3_large",
        epochs=3,
        batch_size=32,
        lr=1e-4,
        seed=42,
        output_dir=ARTIFACT_DIR,
        split_csv_path=SPLIT_CSV,
        images_dir=IMAGES_DIR
    )
    print(f"MobileNetV3 Training finished. Best Val Balanced Acc: {transfer_res['best_val_balanced_acc']:.4f}")

    # 3. Model Selection (Based strictly on validation split performance)
    print("\n--- MODEL SELECTION (VALIDATION SPLIT) ---")
    if transfer_res['best_val_balanced_acc'] >= baseline_res['best_val_balanced_acc']:
        selected_arch = "mobilenetv3_large"
        selected_checkpoint = ARTIFACT_DIR / "mobilenetv3_large_best.pth"
        selected_val_acc = transfer_res['best_val_balanced_acc']
    else:
        selected_arch = "baseline"
        selected_checkpoint = ARTIFACT_DIR / "baseline_best.pth"
        selected_val_acc = baseline_res['best_val_balanced_acc']

    print(f"Selected Model: {selected_arch} (Validation Balanced Acc: {selected_val_acc:.4f})")

    # Copy to production active model checkpoint
    best_model_dest = ARTIFACT_DIR / "best_model.pth"
    shutil.copyfile(selected_checkpoint, best_model_dest)
    print(f"Copied selected model to {best_model_dest}")

    # 4. Final Evaluation on Held-Out Test Split (329 records)
    print("\n--- EVALUATION ON HELD-OUT TEST SPLIT (N=329) ---")
    test_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load checkpoint
    ckpt = torch.load(best_model_dest, map_location=device)
    eval_model = get_model(architecture=selected_arch, num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model.to(device)

    test_metrics = evaluate_model(
        model=eval_model,
        dataloader=test_loader,
        device=device,
        save_dir=ARTIFACT_DIR,
        split_name="test"
    )

    print("\nTEST SPLIT RESULTS:")
    print(f"Overall Accuracy:  {test_metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision:   {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:    {test_metrics['macro_f1']:.4f}")

    print("\nPER-CLASS METRICS:")
    for cls, m in test_metrics['per_class'].items():
        print(f"  {cls:4s} ({m['name']:30s}): F1={m['f1_score']:.4f}, Rec={m['recall']:.4f}, Prec={m['precision']:.4f}, Support={m['support']}")

    # 5. Save model metadata JSON
    metadata = {
        "model_version": f"v1.0.0-{selected_arch}",
        "architecture": selected_arch,
        "classes": DIAGNOSTIC_CLASSES,
        "class_names": CLASS_NAMES,
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_balanced_accuracy": selected_val_acc,
        "test_metrics": test_metrics,
        "baseline_comparison": {
            "baseline_val_balanced_acc": baseline_res['best_val_balanced_acc'],
            "transfer_val_balanced_acc": transfer_res['best_val_balanced_acc'],
        },
        "compute_device": str(device),
        "seed": 42,
        "note": "Melanoma (MEL) has 7 test samples; interpret per-class sensitivity with caution."
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {ARTIFACT_DIR / 'model_metadata.json'}")
    print("=" * 70)
    print("MACHINE LEARNING EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
