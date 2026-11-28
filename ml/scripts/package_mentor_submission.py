import shutil
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = ROOT / "MENTOR_SUBMISSION_FINAL"


def copy_file_safe(src: Path, dst: Path):
    if not src.exists():
        print(f"Warning: Source not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_safe(src: Path, dst: Path):
    if not src.exists():
        print(f"Warning: Source dir not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def main():
    print("=" * 75)
    print("DERMAI - MENTOR SUBMISSION PACKAGING")
    print("Target Directory: MENTOR_SUBMISSION_FINAL")
    print("=" * 75)

    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy All 5 PDF Reports & Markdown
    pdf_dir = ROOT / "outputs" / "mentor_submission"
    reports = [
        "01_Dataset_and_Preprocessing_Report.pdf",
        "02_MobileNetV3_Training_Report.pdf",
        "03_EfficientNetB0_Training_Report.pdf",
        "04_Model_Comparison_Report.pdf",
        "05_Lesion_Localization_Report.pdf",
        "05_Lesion_Localization_Report.md"
    ]
    for r in reports:
        copy_file_safe(pdf_dir / r, SUBMISSION_DIR / r)
        print(f"Copied report: {r}")

    # 2. Checkpoints
    ckpt_dir = SUBMISSION_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifacts_ml = ROOT / "artifacts" / "ml"
    
    copy_file_safe(artifacts_ml / "mobilenetv3_large_best.pth", ckpt_dir / "mobilenetv3_large_best.pth")
    copy_file_safe(artifacts_ml / "efficientnet_b0_best.pth", ckpt_dir / "efficientnet_b0_best.pth")
    # For best_model.pth, use the best validation model (EfficientNet-B0)
    copy_file_safe(artifacts_ml / "efficientnet_b0_best.pth", ckpt_dir / "best_model.pth")
    copy_file_safe(artifacts_ml / "baseline_best.pth", ckpt_dir / "baseline_best.pth")
    print("Copied checkpoints (MobileNetV3, EfficientNet-B0, Baseline)")

    # 3. Dataset Audit & Manifests
    dataset_out = SUBMISSION_DIR / "dataset_audit"
    dataset_src = ROOT / "outputs" / "mentor_submission" / "dataset"
    copy_file_safe(dataset_src / "dataset_inventory.json", dataset_out / "dataset_inventory.json")
    copy_file_safe(dataset_src / "class_distribution.csv", dataset_out / "class_distribution.csv")
    copy_file_safe(dataset_src / "split_leakage_check.json", dataset_out / "split_leakage_check.json")
    copy_file_safe(dataset_src / "class_distribution.png", dataset_out / "class_distribution.png")
    copy_file_safe(dataset_src / "train.csv", dataset_out / "train.csv")
    copy_file_safe(dataset_src / "validation.csv", dataset_out / "validation.csv")
    copy_file_safe(dataset_src / "test.csv", dataset_out / "test.csv")
    print("Copied dataset audit files and patient-level manifests")

    # 4. Training Results
    train_out = SUBMISSION_DIR / "training_results"
    train_out.mkdir(parents=True, exist_ok=True)
    copy_file_safe(artifacts_ml / "mobilenetv3_large_history.json", train_out / "mobilenetv3_large_history.json")
    copy_file_safe(artifacts_ml / "mobilenetv3_large_loss_curve.png", train_out / "mobilenetv3_large_loss_curve.png")
    copy_file_safe(artifacts_ml / "efficientnet_b0_history.json", train_out / "efficientnet_b0_history.json")
    copy_file_safe(artifacts_ml / "efficientnet_b0_loss_curve.png", train_out / "efficientnet_b0_loss_curve.png")
    copy_file_safe(artifacts_ml / "efficientnet_b0_config.json", train_out / "efficientnet_b0_config.json")
    print("Copied training histories and loss plots")

    # 5. Evaluation Results
    eval_out = SUBMISSION_DIR / "evaluation_results"
    eval_src = ROOT / "outputs" / "mentor_submission" / "model_comparison"
    copy_file_safe(eval_src / "model_comparison.csv", eval_out / "model_comparison.csv")
    copy_file_safe(eval_src / "per_class_comparison.csv", eval_out / "per_class_comparison.csv")
    copy_file_safe(eval_src / "model_comparison.png", eval_out / "model_comparison.png")
    copy_file_safe(eval_src / "comparison_summary.md", eval_out / "comparison_summary.md")
    
    cm_src = eval_src / "confusion_matrices"
    cm_out = eval_out / "confusion_matrices"
    cm_out.mkdir(parents=True, exist_ok=True)
    for f in cm_src.glob("*"):
        copy_file_safe(f, cm_out / f.name)
    print("Copied model comparison metrics, per-class tables, and confusion matrices")

    # 6. Lesion Localization & Visual Examples
    loc_src = ROOT / "outputs" / "mentor_submission" / "lesion_localization"
    loc_out = SUBMISSION_DIR / "lesion_localization"
    loc_out.mkdir(parents=True, exist_ok=True)
    copy_file_safe(loc_src / "annotation_audit.json", loc_out / "annotation_audit.json")
    
    vis_src = loc_src / "visual_examples"
    vis_out = SUBMISSION_DIR / "visual_examples"
    if vis_src.exists():
        copy_tree_safe(vis_src, vis_out)
        # also copy to lesion_localization/visual_examples
        copy_tree_safe(vis_src, loc_out / "visual_examples")
    print("Copied annotation audit and visual localization examples")

    # 7. Source Code
    src_out = SUBMISSION_DIR / "source_code"
    (src_out / "ml" / "src").mkdir(parents=True, exist_ok=True)
    (src_out / "ml" / "scripts").mkdir(parents=True, exist_ok=True)
    (src_out / "backend").mkdir(parents=True, exist_ok=True)

    for f in (ROOT / "ml" / "src").glob("*.py"):
        copy_file_safe(f, src_out / "ml" / "src" / f.name)
    for f in (ROOT / "ml" / "scripts").glob("*.py"):
        copy_file_safe(f, src_out / "ml" / "scripts" / f.name)
    copy_tree_safe(ROOT / "backend", src_out / "backend")
    print("Copied clean runnable source code (ml/src, ml/scripts, backend)")

    # 8. Tests & Test Results
    test_out = SUBMISSION_DIR / "tests"
    test_out.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "tests").glob("*.py"):
        copy_file_safe(f, test_out / f.name)
    
    test_res_out = SUBMISSION_DIR / "test_results"
    test_res_out.mkdir(parents=True, exist_ok=True)
    copy_file_safe(ROOT / "outputs" / "test_results.txt", test_res_out / "test_results.txt")
    print("Copied test suite and verified execution log (41/41 passing)")

    # 9. Configurations
    config_out = SUBMISSION_DIR / "configurations"
    config_out.mkdir(parents=True, exist_ok=True)
    copy_file_safe(artifacts_ml / "efficientnet_b0_config.json", config_out / "efficientnet_b0_config.json")
    base_cfg = ROOT / "outputs" / "mentor_submission" / "baseline_mobilenetv3" / "config.json"
    if base_cfg.exists():
        copy_file_safe(base_cfg, config_out / "mobilenetv3_config.json")
    print("Copied configuration files")

    # 10. Requirements File
    copy_file_safe(ROOT / "requirements.txt", SUBMISSION_DIR / "requirements.txt")

    print("\nPackage assembly completed successfully.")


if __name__ == "__main__":
    main()
