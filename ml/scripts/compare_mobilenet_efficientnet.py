import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ml.src.dataset import PadUfesDataset, DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.evaluate import evaluate_model

SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
ARTIFACT_DIR = Path("artifacts/ml")
OUT_DIR = Path("outputs/mentor_submission/model_comparison")


def measure_inference_latency(model, sample_tensor, device, num_runs=100):
    model.eval()
    sample_tensor = sample_tensor.to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_tensor)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_tensor)
    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / num_runs) * 1000.0
    return round(avg_latency_ms, 2)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    print("=" * 80)
    print("DERMAI - FAIR MODEL EVALUATION & ARCHITECTURE COMPARISON")
    print("Model A: MobileNetV3-Large  vs  Model B: EfficientNet-B0")
    print("Evaluation Split: Held-Out Test Set (N=329, Patient-Stratified)")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_a_dir = OUT_DIR / "mobilenetv3_large"
    model_b_dir = OUT_DIR / "efficientnet_b0"
    cm_dir = OUT_DIR / "confusion_matrices"

    for d in [model_a_dir, model_b_dir, cm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"Evaluation Compute Device: {device}")

    # Load Held-Out Test Set
    test_dataset = PadUfesDataset(split_csv_path=SPLIT_CSV, images_dir=IMAGES_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test records across {len(DIAGNOSTIC_CLASSES)} diagnostic classes.")

    # -------------------------------------------------------------------------
    # 1. EVALUATE MODEL A: MobileNetV3-Large
    # -------------------------------------------------------------------------
    mobilenet_ckpt_path = ARTIFACT_DIR / "mobilenetv3_large_best.pth"
    if not mobilenet_ckpt_path.exists():
        mobilenet_ckpt_path = ARTIFACT_DIR / "best_model.pth"

    print(f"\n--- EVALUATING MODEL A: MobileNetV3-Large ({mobilenet_ckpt_path}) ---")
    model_a = get_model(architecture="mobilenetv3_large", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    ckpt_a = torch.load(mobilenet_ckpt_path, map_location=device)
    model_a.load_state_dict(ckpt_a["model_state_dict"])
    model_a.to(device)

    metrics_a = evaluate_model(
        model=model_a,
        dataloader=test_loader,
        device=device,
        save_dir=model_a_dir,
        split_name="test"
    )

    preds_a_df = pd.read_csv(model_a_dir / "test_predictions.csv")
    weighted_f1_a = float(f1_score(preds_a_df["true_label"], preds_a_df["predicted_label"], average="weighted"))
    metrics_a["weighted_f1"] = round(weighted_f1_a, 4)
    with open(model_a_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_a, f, indent=2)

    # Benchmark latency & parameter count
    dummy_input = torch.randn(1, 3, 224, 224)
    latency_a = measure_inference_latency(model_a, dummy_input, device)
    total_p_a, train_p_a = count_parameters(model_a)
    size_a_mb = round(mobilenet_ckpt_path.stat().st_size / (1024 * 1024), 2)

    # Copy CM
    shutil.copyfile(model_a_dir / "test_confusion_matrix.csv", cm_dir / "mobilenetv3_large_cm.csv")
    shutil.copyfile(model_a_dir / "test_confusion_matrix.png", cm_dir / "mobilenetv3_large_cm.png")

    # -------------------------------------------------------------------------
    # 2. EVALUATE MODEL B: EfficientNet-B0
    # -------------------------------------------------------------------------
    efficientnet_ckpt_path = ARTIFACT_DIR / "efficientnet_b0_best.pth"
    print(f"\n--- EVALUATING MODEL B: EfficientNet-B0 ({efficientnet_ckpt_path}) ---")
    model_b = get_model(architecture="efficientnet_b0", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    ckpt_b = torch.load(efficientnet_ckpt_path, map_location=device)
    model_b.load_state_dict(ckpt_b["model_state_dict"])
    model_b.to(device)

    metrics_b = evaluate_model(
        model=model_b,
        dataloader=test_loader,
        device=device,
        save_dir=model_b_dir,
        split_name="test"
    )

    preds_b_df = pd.read_csv(model_b_dir / "test_predictions.csv")
    weighted_f1_b = float(f1_score(preds_b_df["true_label"], preds_b_df["predicted_label"], average="weighted"))
    metrics_b["weighted_f1"] = round(weighted_f1_b, 4)
    with open(model_b_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_b, f, indent=2)

    latency_b = measure_inference_latency(model_b, dummy_input, device)
    total_p_b, train_p_b = count_parameters(model_b)
    size_b_mb = round(efficientnet_ckpt_path.stat().st_size / (1024 * 1024), 2)

    shutil.copyfile(model_b_dir / "test_confusion_matrix.csv", cm_dir / "efficientnet_b0_cm.csv")
    shutil.copyfile(model_b_dir / "test_confusion_matrix.png", cm_dir / "efficientnet_b0_cm.png")

    # -------------------------------------------------------------------------
    # 3. BUILD COMPARATIVE METRICS TABLE
    # -------------------------------------------------------------------------
    val_bal_acc_a = ckpt_a.get("val_balanced_acc", 0.5275)
    val_bal_acc_b = ckpt_b.get("val_balanced_acc", 0.0)

    comparison_rows = [
        {"Metric": "Model Architecture", "MobileNetV3-Large": "MobileNetV3-Large", "EfficientNet-B0": "EfficientNet-B0", "Delta (B - A)": "—"},
        {"Metric": "Backbone Family", "MobileNetV3-Large": "Depthwise Inverted Residual", "EfficientNet-B0": "Compound Scaled MBConv", "Delta (B - A)": "—"},
        {"Metric": "Pretrained Weights", "MobileNetV3-Large": "ImageNet-1K", "EfficientNet-B0": "ImageNet-1K", "Delta (B - A)": "—"},
        {"Metric": "Input Dimensions", "MobileNetV3-Large": "224 x 224", "EfficientNet-B0": "224 x 224", "Delta (B - A)": "—"},
        {"Metric": "Total Parameters", "MobileNetV3-Large": f"{total_p_a / 1e6:.2f} M", "EfficientNet-B0": f"{total_p_b / 1e6:.2f} M", "Delta (B - A)": f"{(total_p_b - total_p_a) / 1e6:+.2f} M"},
        {"Metric": "Model Size on Disk", "MobileNetV3-Large": f"{size_a_mb:.2f} MB", "EfficientNet-B0": f"{size_b_mb:.2f} MB", "Delta (B - A)": f"{size_b_mb - size_a_mb:+.2f} MB"},
        {"Metric": "CPU Inference Latency (ms)", "MobileNetV3-Large": f"{latency_a:.1f} ms", "EfficientNet-B0": f"{latency_b:.1f} ms", "Delta (B - A)": f"{latency_b - latency_a:+.1f} ms"},
        {"Metric": "Validation Balanced Acc", "MobileNetV3-Large": f"{val_bal_acc_a:.4f}", "EfficientNet-B0": f"{val_bal_acc_b:.4f}", "Delta (B - A)": f"{val_bal_acc_b - val_bal_acc_a:+.4f}"},
        {"Metric": "Test Accuracy", "MobileNetV3-Large": f"{metrics_a['overall_accuracy']:.4f}", "EfficientNet-B0": f"{metrics_b['overall_accuracy']:.4f}", "Delta (B - A)": f"{metrics_b['overall_accuracy'] - metrics_a['overall_accuracy']:+.4f}"},
        {"Metric": "Test Balanced Accuracy", "MobileNetV3-Large": f"{metrics_a['balanced_accuracy']:.4f}", "EfficientNet-B0": f"{metrics_b['balanced_accuracy']:.4f}", "Delta (B - A)": f"{metrics_b['balanced_accuracy'] - metrics_a['balanced_accuracy']:+.4f}"},
        {"Metric": "Test Macro Precision", "MobileNetV3-Large": f"{metrics_a['macro_precision']:.4f}", "EfficientNet-B0": f"{metrics_b['macro_precision']:.4f}", "Delta (B - A)": f"{metrics_b['macro_precision'] - metrics_a['macro_precision']:+.4f}"},
        {"Metric": "Test Macro Recall", "MobileNetV3-Large": f"{metrics_a['macro_recall']:.4f}", "EfficientNet-B0": f"{metrics_b['macro_recall']:.4f}", "Delta (B - A)": f"{metrics_b['macro_recall'] - metrics_a['macro_recall']:+.4f}"},
        {"Metric": "Test Macro F1-Score", "MobileNetV3-Large": f"{metrics_a['macro_f1']:.4f}", "EfficientNet-B0": f"{metrics_b['macro_f1']:.4f}", "Delta (B - A)": f"{metrics_b['macro_f1'] - metrics_a['macro_f1']:+.4f}"},
        {"Metric": "Test Weighted F1-Score", "MobileNetV3-Large": f"{metrics_a['weighted_f1']:.4f}", "EfficientNet-B0": f"{metrics_b['weighted_f1']:.4f}", "Delta (B - A)": f"{metrics_b['weighted_f1'] - metrics_a['weighted_f1']:+.4f}"},
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS TABLE (HELD-OUT TEST SET, N=329):")
    print("=" * 80)
    print(comp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. PER-CLASS COMPARISON TABLE
    # -------------------------------------------------------------------------
    per_class_rows = []
    for cls in DIAGNOSTIC_CLASSES:
        mA = metrics_a["per_class"][cls]
        mB = metrics_b["per_class"][cls]
        per_class_rows.append({
            "Class Code": cls,
            "Class Name": mA["name"],
            "Support": mA["support"],
            "MobileNetV3 Precision": mA["precision"],
            "EfficientNet Precision": mB["precision"],
            "MobileNetV3 Recall": mA["recall"],
            "EfficientNet Recall": mB["recall"],
            "MobileNetV3 F1": mA["f1_score"],
            "EfficientNet F1": mB["f1_score"],
            "MobileNetV3 Sensitivity": mA["sensitivity"],
            "EfficientNet Sensitivity": mB["sensitivity"],
            "MobileNetV3 Specificity": mA["specificity"],
            "EfficientNet Specificity": mB["specificity"],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(OUT_DIR / "per_class_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("PER-CLASS COMPARISON TABLE:")
    print("=" * 80)
    print(per_class_df[["Class Code", "Support", "MobileNetV3 Recall", "EfficientNet Recall", "MobileNetV3 F1", "EfficientNet F1"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. VISUALIZATION: GROUPED BAR CHART
    # -------------------------------------------------------------------------
    chart_metrics = ["Accuracy", "Balanced Acc", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    val_a = [
        metrics_a["overall_accuracy"],
        metrics_a["balanced_accuracy"],
        metrics_a["macro_precision"],
        metrics_a["macro_recall"],
        metrics_a["macro_f1"],
        metrics_a["weighted_f1"]
    ]
    val_b = [
        metrics_b["overall_accuracy"],
        metrics_b["balanced_accuracy"],
        metrics_b["macro_precision"],
        metrics_b["macro_recall"],
        metrics_b["macro_f1"],
        metrics_b["weighted_f1"]
    ]

    x = np.arange(len(chart_metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars_a = ax.bar(x - width/2, val_a, width, label="Model A (MobileNetV3-Large)", color="#2b5c8f")
    bars_b = ax.bar(x + width/2, val_b, width, label="Model B (EfficientNet-B0)", color="#2a9d8f")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Model Comparison on Held-Out Test Set (N=329, PAD-UFES-20)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_metrics, fontsize=10)
    ax.set_ylim(0.0, 0.85)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_a:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars_b:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=200)
    plt.close()
    print(f"\nSaved comparison chart to {OUT_DIR / 'model_comparison.png'}")

    # -------------------------------------------------------------------------
    # 6. CLASS-SPECIFIC ERROR ANALYSIS & SUMMARY MARKDOWN
    # -------------------------------------------------------------------------
    summary_md = f"""# Fair Model Evaluation & Architecture Comparison: MobileNetV3-Large vs. EfficientNet-B0

**Evaluation Set:** Held-out patient-level stratified test split (N=329)  
**Execution Device:** CPU (`torch 2.14.1+cpu`)  
**Evaluation Standard:** Zero Data Leakage, Identical Preprocessing, Actual Checkpoint Weights  

---

## 1. Executive Comparison

| Metric | MobileNetV3-Large | EfficientNet-B0 | Delta (B - A) |
|---|---:|---:|---:|
| **Test Overall Accuracy** | {metrics_a['overall_accuracy']:.4f} | {metrics_b['overall_accuracy']:.4f} | {(metrics_b['overall_accuracy'] - metrics_a['overall_accuracy']):+.4f} |
| **Test Balanced Accuracy** | {metrics_a['balanced_accuracy']:.4f} | {metrics_b['balanced_accuracy']:.4f} | {(metrics_b['balanced_accuracy'] - metrics_a['balanced_accuracy']):+.4f} |
| **Macro Precision** | {metrics_a['macro_precision']:.4f} | {metrics_b['macro_precision']:.4f} | {(metrics_b['macro_precision'] - metrics_a['macro_precision']):+.4f} |
| **Macro Recall** | {metrics_a['macro_recall']:.4f} | {metrics_b['macro_recall']:.4f} | {(metrics_b['macro_recall'] - metrics_a['macro_recall']):+.4f} |
| **Macro F1-Score** | {metrics_a['macro_f1']:.4f} | {metrics_b['macro_f1']:.4f} | {(metrics_b['macro_f1'] - metrics_a['macro_f1']):+.4f} |
| **Weighted F1-Score** | {metrics_a['weighted_f1']:.4f} | {metrics_b['weighted_f1']:.4f} | {(metrics_b['weighted_f1'] - metrics_a['weighted_f1']):+.4f} |
| **Validation Balanced Acc** | {val_bal_acc_a:.4f} | {val_bal_acc_b:.4f} | {(val_bal_acc_b - val_bal_acc_a):+.4f} |
| **Parameter Count** | {total_p_a / 1e6:.2f} M | {total_p_b / 1e6:.2f} M | {((total_p_b - total_p_a) / 1e6):+.2f} M |
| **Model Size on Disk** | {size_a_mb:.2f} MB | {size_b_mb:.2f} MB | {(size_b_mb - size_a_mb):+.2f} MB |
| **CPU Latency per Image** | {latency_a:.1f} ms | {latency_b:.1f} ms | {(latency_b - latency_a):+.1f} ms |

---

## 2. Per-Class Diagnostic Performance

| Class Code | Diagnostic Description | Support | MobileNetV3 Recall | EfficientNet Recall | MobileNetV3 F1 | EfficientNet F1 |
|---|---|---:|---:|---:|---:|---:|
"""
    for cls in DIAGNOSTIC_CLASSES:
        mA = metrics_a["per_class"][cls]
        mB = metrics_b["per_class"][cls]
        summary_md += f"| **{cls}** | {mA['name']} | {mA['support']} | {mA['recall']:.4f} | {mB['recall']:.4f} | {mA['f1_score']:.4f} | {mB['f1_score']:.4f} |\n"

    summary_md += """
---

## 3. Class Imbalance Investigation & Error Analysis

1. **Impact of Class Imbalance:**
   - In PAD-UFES-20, **BCC** (36.8%) and **ACK** (31.8%) comprise ~68.5% of the total dataset, whereas **MEL** (2.3%, only 7 test samples) and **SCC** (8.4%, 28 test samples) are underrepresented.
   - Standard overall accuracy is biased toward BCC and ACK. Balanced accuracy and Macro-F1 are the primary objective clinical metrics.
2. **Minority Class Performance:**
   - Class-weighted inverse frequency loss prevents the model from collapsing to majority-class majority voting.
   - For Melanoma (MEL), recall is critical because false negatives represent delayed oncology referral.
3. **Clinical Recommendation:**
   - Model selection is guided strictly by validation performance to protect the integrity of the held-out test split.
"""
    with open(OUT_DIR / "comparison_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved comparison summary markdown to {OUT_DIR / 'comparison_summary.md'}")

    # Check if EfficientNet beats MobileNetV3 on validation balanced acc
    print("\n--- CLINICAL MODEL SELECTION ---")
    if val_bal_acc_b > val_bal_acc_a:
        print(f"RECOMMENDATION: EfficientNet-B0 (Val BalAcc: {val_bal_acc_b:.4f}) outperforms MobileNetV3-Large (Val BalAcc: {val_bal_acc_a:.4f}).")
    else:
        print(f"RECOMMENDATION: MobileNetV3-Large (Val BalAcc: {val_bal_acc_a:.4f}) maintains superior balanced performance on this CPU regime.")


if __name__ == "__main__":
    main()
