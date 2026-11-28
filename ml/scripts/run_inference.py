import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.preprocessing import load_image_safely, preprocess_image
from ml.src.explainability import GradCAM
from backend.app.services.segmentation_service import ClassicalLesionLocalizer

DEFAULT_CHECKPOINT = Path("artifacts/ml/best_model.pth")
DEFAULT_EFFICIENTNET = Path("artifacts/ml/efficientnet_b0_best.pth")
DEFAULT_MOBILENET = Path("artifacts/ml/mobilenetv3_large_best.pth")


def resolve_image_path(image_input: str) -> Path:
    p = Path(image_input)
    if p.exists() and p.is_file():
        return p

    # Search in dataset directory
    dataset_dir = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
    if dataset_dir.exists():
        matches = list(dataset_dir.rglob(f"*{p.name}"))
        if matches:
            return matches[0]

    # Search in fixtures
    fixtures_dir = Path("artifacts/test_fixtures")
    if fixtures_dir.exists():
        matches = list(fixtures_dir.rglob(f"*{p.name}"))
        if matches:
            return matches[0]

    return p


def run_inference(
    image_path: Path,
    architecture: str = "auto",
    checkpoint_path: Path = None,
    output_dir: Path = None,
    generate_gradcam: bool = True
):
    image_path = resolve_image_path(str(image_path))
    if not image_path.exists():
        print(f"ERROR: Image file not found at: {image_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load image safely
    try:
        rgb_image = load_image_safely(image_path)
    except Exception as e:
        print(f"ERROR: Failed to load image safely. Image may be corrupt or invalid: {e}", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cpu")

    # 2. Determine checkpoint and architecture
    if checkpoint_path is None:
        if architecture == "efficientnet_b0":
            checkpoint_path = DEFAULT_EFFICIENTNET
        elif architecture == "mobilenetv3_large":
            checkpoint_path = DEFAULT_MOBILENET
        else:
            checkpoint_path = DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else DEFAULT_MOBILENET

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Model checkpoint not found at: {checkpoint_path}", file=sys.stderr)
        sys.exit(3)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt.get("architecture", architecture if architecture != "auto" else "mobilenetv3_large")
    val_bal_acc = ckpt.get("val_balanced_acc", "N/A")
    epoch = ckpt.get("epoch", "N/A")

    # Instantiate model
    model = get_model(architecture=arch, num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Preprocess
    _, input_tensor = preprocess_image(rgb_image, target_size=(224, 224))
    input_tensor = input_tensor.to(device)

    # 4. Model Forward Pass
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probabilities.argmax())
    predicted_class = DIAGNOSTIC_CLASSES[pred_idx]
    confidence_score = float(probabilities[pred_idx])

    # 5. Otsu ROI Proposal
    np_img = np.asarray(rgb_image)
    localizer = ClassicalLesionLocalizer()
    mask, roi_data = localizer.localize_roi(np_img)

    # 6. Grad-CAM Overlay
    overlay_path = None
    if generate_gradcam and hasattr(model, "get_target_layer_for_gradcam"):
        try:
            target_layer = model.get_target_layer_for_gradcam()
            tensor_grad = input_tensor.clone().requires_grad_(True)
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate_heatmap(tensor_grad, target_class_idx=pred_idx)
            overlay = gradcam.create_overlay(rgb_image, heatmap, alpha=0.5)
            gradcam.remove_hooks()

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = output_dir / f"{image_path.stem}_{arch}_gradcam.png"
                overlay.save(overlay_path)
                mask_path = output_dir / f"{image_path.stem}_otsu_mask.png"
                Image.fromarray(mask).save(mask_path)
        except Exception as e:
            print(f"Warning: Grad-CAM generation encountered an issue: {e}", file=sys.stderr)

    # Print Clean CLI Output
    print("=" * 65)
    print("DERMAI - SKIN LESION CLASSIFICATION & LOCALIZATION INFERENCE")
    print("=" * 65)
    print(f"Input Image:         {image_path.name}")
    print(f"Image Resolution:    {rgb_image.size[0]} x {rgb_image.size[1]} ({rgb_image.mode})")
    print(f"Loaded Architecture: {arch}")
    print(f"Checkpoint File:     {checkpoint_path}")
    print(f"Checkpoint Epoch:    {epoch}")
    print(f"Val Balanced Acc:    {val_bal_acc if isinstance(val_bal_acc, str) else f'{val_bal_acc:.4f}'}")
    print(f"Execution Device:    {device}")
    print("-" * 65)
    print(f"PREDICTED DIAGNOSIS: {predicted_class} - {CLASS_NAMES.get(predicted_class, predicted_class)}")
    print(f"CONFIDENCE SCORE:    {confidence_score * 100:.2f}%")
    print("-" * 65)
    print("CLASS PROBABILITY DISTRIBUTION:")
    sorted_indices = probabilities.argsort()[::-1]
    for idx in sorted_indices:
        cls_code = DIAGNOSTIC_CLASSES[idx]
        p = float(probabilities[idx])
        bar = "#" * int(p * 30)
        marker = " <== PREDICTED" if idx == pred_idx else ""
        print(f"  {cls_code:4s} ({CLASS_NAMES[cls_code]:27s}): {p*100:6.2f}% {bar}{marker}")

    print("-" * 65)
    print(f"Otsu Candidate ROI:  BBox [x={roi_data['bbox'][0]}, y={roi_data['bbox'][1]}, w={roi_data['bbox'][2]}, h={roi_data['bbox'][3]}]")
    print(f"ROI Area Ratio:      {roi_data['roi_area_ratio']:.3f} of total image area")
    if overlay_path:
        print(f"Grad-CAM Heatmap:    {overlay_path}")
    print("=" * 65)

    return {
        "image": str(image_path),
        "architecture": arch,
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES.get(predicted_class, predicted_class),
        "confidence": round(confidence_score, 4),
        "probabilities": {DIAGNOSTIC_CLASSES[i]: round(float(probabilities[i]), 4) for i in range(len(DIAGNOSTIC_CLASSES))},
        "roi_bbox": roi_data["bbox"],
        "roi_area_ratio": roi_data["roi_area_ratio"],
        "gradcam_path": str(overlay_path) if overlay_path else None
    }


def main():
    parser = argparse.ArgumentParser(description="DermAI Skin Disease Diagnostic Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input clinical skin image")
    parser.add_argument("--architecture", type=str, default="auto", choices=["auto", "mobilenetv3_large", "efficientnet_b0"], help="Backbone architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save ROI and Grad-CAM visualizations")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap generation")

    args = parser.parse_args()
    run_inference(
        image_path=Path(args.image),
        architecture=args.architecture,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        generate_gradcam=not args.no_gradcam
    )


if __name__ == "__main__":
    main()
