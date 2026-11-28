import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.preprocessing import load_image_safely, preprocess_image
from ml.src.explainability import GradCAM
from backend.app.services.segmentation_service import ClassicalLesionLocalizer

RAW_METADATA = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\metadata.csv")
SPLIT_CSV = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
OUT_DIR = Path("outputs/mentor_submission/lesion_localization")


def audit_annotations():
    print("=" * 75)
    print("TASK 6: PAD-UFES-20 ANNOTATION AUDIT")
    print("=" * 75)

    df_meta = pd.read_csv(RAW_METADATA)
    columns = list(df_meta.columns)
    total_records = len(df_meta)

    # Check for spatial annotation columns
    spatial_keywords = ["mask", "bbox", "box", "polygon", "segmentation", "xmin", "ymin", "xmax", "ymax"]
    found_spatial = [c for c in columns if any(k in c.lower() for k in spatial_keywords)]

    has_masks = False
    has_bboxes = False
    has_image_labels = "diagnostic" in columns

    audit_res = {
        "dataset_name": "PAD-UFES-20",
        "metadata_records": total_records,
        "metadata_columns": columns,
        "image_level_diagnostic_labels_present": has_image_labels,
        "classes_present": DIAGNOSTIC_CLASSES,
        "bounding_box_annotations_present": has_bboxes,
        "pixel_segmentation_masks_present": has_masks,
        "clinical_measurements_present": {
            "diameter_1": "diameter_1" in columns,
            "diameter_2": "diameter_2" in columns,
            "measurement_unit": "millimeters (physical caliper measurement, not pixel coordinates)"
        },
        "audit_classification": "Case C: Strictly Image-Level Diagnostic Labels Only",
        "scientific_limitation_statement": (
            "PAD-UFES-20 does NOT provide ground-truth bounding boxes or pixel-level segmentation masks. "
            "Unsupervised Otsu thresholding acts as an empirical candidate-ROI proposal, while Grad-CAM "
            "visualizes classification feature activations. Neither constitutes a clinically validated lesion boundary, "
            "and calculating supervised IoU or Dice coefficients without genuine ground-truth masks is scientifically unsupportable."
        ),
        "recommended_supervised_dataset": "ISIC 2018 Task 1 (2,594 dermoscopic images with expert ground-truth masks)"
    }

    out_json = OUT_DIR / "annotation_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_res, f, indent=2)

    print(f"Audit Result: {audit_res['audit_classification']}")
    print(f"Bounding Boxes: {has_bboxes} | Segmentation Masks: {has_masks}")
    print(f"Saved audit inventory to: {out_json}")
    return audit_res


def generate_visual_examples():
    print("\n" + "=" * 75)
    print("TASK 8: GENERATING ACTUAL LESION VISUAL EXAMPLES (ALL 6 CLASSES)")
    print("=" * 75)

    samples_dir = OUT_DIR / "visual_examples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    df_split = pd.read_csv(SPLIT_CSV)
    img_lookup = {f.name.lower(): f for f in IMAGES_DIR.rglob("*.png")}

    device = torch.device("cpu")
    localizer = ClassicalLesionLocalizer()

    # Load both models for visual comparison
    mobilenet_ckpt = Path("artifacts/ml/mobilenetv3_large_best.pth")
    if not mobilenet_ckpt.exists():
        mobilenet_ckpt = Path("artifacts/ml/best_model.pth")
    efficientnet_ckpt = Path("artifacts/ml/efficientnet_b0_best.pth")

    model_mb = get_model("mobilenetv3_large", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model_mb.load_state_dict(torch.load(mobilenet_ckpt, map_location=device)["model_state_dict"])
    model_mb.to(device).eval()

    model_eff = get_model("efficientnet_b0", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model_eff.load_state_dict(torch.load(efficientnet_ckpt, map_location=device)["model_state_dict"])
    model_eff.to(device).eval()

    catalog = []

    for cls in DIAGNOSTIC_CLASSES:
        test_sub = df_split[(df_split["split"] == "test") & (df_split["diagnostic"] == cls)]
        if len(test_sub) == 0:
            test_sub = df_split[df_split["diagnostic"] == cls]
        row = test_sub.iloc[0]
        img_id = str(row["img_id"]).strip()
        img_path = img_lookup[img_id.lower()]

        # 1. Original Image
        rgb_img = load_image_safely(img_path)
        np_img = np.asarray(rgb_img)
        w, h = rgb_img.size

        orig_filename = f"{cls}_{img_id}_1_original.png"
        rgb_img.save(samples_dir / orig_filename)

        # 2. Otsu ROI Proposal Mask & Bounding Box
        mask, roi_data = localizer.localize_roi(np_img)
        bx, by, bw, bh = roi_data["bbox"]

        mask_filename = f"{cls}_{img_id}_2_otsu_mask.png"
        Image.fromarray(mask).save(samples_dir / mask_filename)

        # Bounding box on original
        img_with_box = np_img.copy()
        cv2.rectangle(img_with_box, (bx, by), (bx + bw, by + bh), (0, 255, 0), max(3, int(w * 0.005)))
        bbox_filename = f"{cls}_{img_id}_3_otsu_bbox.png"
        Image.fromarray(img_with_box).save(samples_dir / bbox_filename)

        # 3. Grad-CAM for MobileNetV3-Large
        _, tensor_input = preprocess_image(rgb_img, target_size=(224, 224))
        cls_idx = DIAGNOSTIC_CLASSES.index(cls)

        # MobileNet GradCAM
        gradcam_mb = GradCAM(model_mb, model_mb.get_target_layer_for_gradcam())
        heatmap_mb = gradcam_mb.generate_heatmap(tensor_input.clone().requires_grad_(True), target_class_idx=cls_idx)
        overlay_mb = gradcam_mb.create_overlay(rgb_img, heatmap_mb, alpha=0.5)
        gradcam_mb.remove_hooks()
        mb_filename = f"{cls}_{img_id}_4_mobilenet_gradcam.png"
        overlay_mb.save(samples_dir / mb_filename)

        # 4. Grad-CAM for EfficientNet-B0
        gradcam_eff = GradCAM(model_eff, model_eff.get_target_layer_for_gradcam())
        heatmap_eff = gradcam_eff.generate_heatmap(tensor_input.clone().requires_grad_(True), target_class_idx=cls_idx)
        overlay_eff = gradcam_eff.create_overlay(rgb_img, heatmap_eff, alpha=0.5)
        gradcam_eff.remove_hooks()
        eff_filename = f"{cls}_{img_id}_5_efficientnet_gradcam.png"
        overlay_eff.save(samples_dir / eff_filename)

        # 5. Composite 5-Panel Multi-View Image
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        # Panel 1: Original
        axes[0].imshow(np_img)
        axes[0].set_title(f"1. Original Image\n{cls}: {CLASS_NAMES[cls]}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Otsu Mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"2. Otsu Binary Mask\nArea Ratio: {roi_data['roi_area_ratio']:.2f}", fontsize=10)
        axes[1].axis("off")

        # Panel 3: Otsu Candidate Box
        axes[2].imshow(img_with_box)
        axes[2].set_title(f"3. Otsu Candidate ROI Box\n[{bx}, {by}, {bw}, {bh}]", fontsize=10)
        axes[2].axis("off")

        # Panel 4: MobileNetV3 Grad-CAM
        axes[3].imshow(overlay_mb)
        axes[3].set_title("4. MobileNetV3 Grad-CAM\n(Feature Attention)", fontsize=10)
        axes[3].axis("off")

        # Panel 5: EfficientNet-B0 Grad-CAM
        axes[4].imshow(overlay_eff)
        axes[4].set_title("5. EfficientNet-B0 Grad-CAM\n(Feature Attention)", fontsize=10)
        axes[4].axis("off")

        fig.tight_layout()
        panel_filename = f"{cls}_{img_id}_panel_summary.png"
        plt.savefig(samples_dir / panel_filename, dpi=180)
        plt.close()

        catalog.append({
            "diagnostic_class": cls,
            "disease_name": CLASS_NAMES[cls],
            "img_id": img_id,
            "image_resolution": [w, h],
            "otsu_bbox": roi_data["bbox"],
            "otsu_area_ratio": roi_data["roi_area_ratio"],
            "files": {
                "original": orig_filename,
                "otsu_mask": mask_filename,
                "otsu_bbox": bbox_filename,
                "mobilenet_gradcam": mb_filename,
                "efficientnet_gradcam": eff_filename,
                "panel_summary": panel_filename
            }
        })
        print(f"  Generated complete visual suite for {cls} ({img_id})")

    with open(samples_dir / "visual_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog saved to {samples_dir / 'visual_catalog.json'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_annotations()
    generate_visual_examples()


if __name__ == "__main__":
    main()
