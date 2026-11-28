import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch

from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.preprocessing import load_image_safely, preprocess_image
from ml.src.explainability import GradCAM
from backend.app.services.segmentation_service import ClassicalLesionLocalizer


def main():
    split_csv = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
    images_dir = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
    ckpt_path = Path("artifacts/ml/best_model.pth")
    out_dir = Path("outputs/mentor_submission/lesion_detection/prototype_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    img_lookup = {f.name.lower(): f for f in images_dir.rglob("*.png")}

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(architecture="mobilenetv3_large", num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    localizer = ClassicalLesionLocalizer()
    target_layer = model.get_target_layer_for_gradcam()

    prototype_records = []

    print("Generating lesion detection and localization prototype samples...")
    for cls in DIAGNOSTIC_CLASSES:
        sub = df[df["diagnostic"] == cls]
        row = sub.iloc[0]
        img_id = str(row["img_id"]).strip()
        img_path = img_lookup[img_id.lower()]

        # 1. Original Image
        rgb_img = load_image_safely(img_path)
        np_img = np.asarray(rgb_img)
        orig_filename = f"{cls}_{img_id}_original.png"
        rgb_img.save(out_dir / orig_filename)

        # 2. Classical Otsu ROI Mask & Bounding Box
        mask, roi_data = localizer.localize_roi(np_img)
        mask_filename = f"{cls}_{img_id}_otsu_mask.png"
        Image.fromarray(mask).save(out_dir / mask_filename)

        # 3. Deep Learning Grad-CAM Attention Overlay
        _, tensor = preprocess_image(rgb_img, target_size=(224, 224))
        tensor_grad = tensor.clone().requires_grad_(True)
        gradcam = GradCAM(model, target_layer)
        cls_idx = DIAGNOSTIC_CLASSES.index(cls)
        heatmap = gradcam.generate_heatmap(tensor_grad, target_class_idx=cls_idx)
        overlay = gradcam.create_overlay(rgb_img, heatmap, alpha=0.5)
        gradcam.remove_hooks()
        gradcam_filename = f"{cls}_{img_id}_gradcam.png"
        overlay.save(out_dir / gradcam_filename)

        # 4. Create 3-Panel Comparative Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original with Otsu BBox
        img_with_bbox = np_img.copy()
        bx, by, bw, bh = roi_data["bbox"]
        cv2.rectangle(img_with_bbox, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
        axes[0].imshow(img_with_bbox)
        axes[0].set_title(f"Original + Otsu ROI Box\n{cls} ({CLASS_NAMES[cls]})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Otsu Binary Mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"Unsupervised Otsu Mask (CIE-Lab)\nArea Ratio: {roi_data['roi_area_ratio']}", fontsize=11)
        axes[1].axis("off")

        # Panel 3: Grad-CAM Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Weakly-Supervised Grad-CAM\n(Model Receptive Attention)", fontsize=11)
        axes[2].axis("off")

        fig.tight_layout()
        panel_filename = f"{cls}_{img_id}_panel.png"
        plt.savefig(out_dir / panel_filename, dpi=180)
        plt.close()

        prototype_records.append({
            "diagnostic_class": cls,
            "class_name": CLASS_NAMES[cls],
            "img_id": img_id,
            "image_size": list(rgb_img.size),
            "otsu_roi_bbox": roi_data["bbox"],
            "otsu_area_ratio": roi_data["roi_area_ratio"],
            "files": {
                "original": orig_filename,
                "otsu_mask": mask_filename,
                "gradcam_overlay": gradcam_filename,
                "composite_panel": panel_filename
            }
        })

    with open(out_dir / "prototype_index.json", "w", encoding="utf-8") as f:
        json.dump(prototype_records, f, indent=2)

    print(f"Successfully generated prototype samples for all 6 classes in {out_dir}!")


if __name__ == "__main__":
    main()
