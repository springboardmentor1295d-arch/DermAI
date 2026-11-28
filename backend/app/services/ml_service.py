import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone
import json
import torch
import torch.nn as nn

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.core.logging import logger
from backend.app.models.model_version import ModelVersion
from backend.app.schemas.prediction import TopNItem, Top3PredictionItem
from backend.app.services.image_processing import ImageProcessingService
from backend.app.services.image_quality import ImageQualityService
from backend.app.services.segmentation_service import LesionSegmentationService
from backend.app.services.risk_service import RiskStratificationService
from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model
from ml.src.explainability import GradCAM
from ml.src.calibration import TemperatureScaling


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "full_name": "DermAI EfficientNet-B0 Classifier",
        "default_version_tag": "v1.1.0-efficientnet_b0",
        "weights_path": settings.ML_ARTIFACT_DIR / "efficientnet_b0_best.pth",
        "calibration_path": settings.ML_ARTIFACT_DIR / "temperature_scaling.json",
        "metadata_path": settings.ML_ARTIFACT_DIR / "model_metadata.json",
    },
    "mobilenetv3_large": {
        "display_name": "MobileNetV3-Large",
        "full_name": "DermAI MobileNetV3-Large Classifier",
        "default_version_tag": "v1.1.0-mobilenetv3_large",
        "weights_path": settings.ML_ARTIFACT_DIR / "mobilenetv3_large_best.pth",
        "calibration_path": settings.ML_ARTIFACT_DIR / "mobilenetv3_temperature_scaling.json",
        "metadata_path": settings.ML_ARTIFACT_DIR / "mobilenetv3_metadata.json",
    },
}


def normalize_architecture_key(arch_or_name: Optional[str]) -> str:
    raw = (arch_or_name or "").strip().lower()
    if "mobilenet" in raw:
        return "mobilenetv3_large"
    return "efficientnet_b0"


class MLInferenceService:
    """
    Singleton ML Inference Service supporting runtime switching between
    verified production models (EfficientNet-B0 and MobileNetV3-Large),
    model-specific post-hoc Temperature Scaling calibration, and Grad-CAM explainability.
    """

    _instance = None
    _lock = threading.Lock()
    model: Optional[nn.Module] = None
    calibrator: Optional[TemperatureScaling] = None
    device: torch.device = torch.device("cpu")
    model_version_tag: str = "v1.1.0-efficientnet_b0"
    architecture: str = "EfficientNet-B0"
    architecture_key: str = "efficientnet_b0"
    active_weights_path: Optional[Path] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance

    def _resolve_active_model_record(self) -> Optional[ModelVersion]:
        db = SessionLocal()
        try:
            return db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        except Exception as exc:
            logger.warning(f"Could not read active ModelVersion from DB during init: {exc}")
            return None
        finally:
            db.close()

    def _resolve_paths_for_arch(
        self, arch_key: str, db_weights_path: Optional[str] = None
    ) -> Tuple[Path, Path]:
        spec = MODEL_SPECS[arch_key]
        canonical_weights = Path(spec["weights_path"])
        calib_path = Path(spec["calibration_path"])

        if db_weights_path:
            candidate = Path(db_weights_path)
            if not candidate.is_absolute():
                candidate = settings.BASE_DIR / candidate
            if candidate.exists() and candidate.name != "best_model.pth":
                return candidate, calib_path

        return canonical_weights, calib_path

    def initialize(self, model_version: Optional[ModelVersion] = None):
        if model_version is None:
            model_version = self._resolve_active_model_record()

        if model_version is not None:
            self.load_model_version(
                architecture=model_version.architecture,
                version_tag=model_version.version_tag,
                weights_path=model_version.weights_path,
            )
        else:
            self.load_model_version(
                architecture="EfficientNet-B0",
                version_tag="v1.1.0-efficientnet_b0",
                weights_path=str(MODEL_SPECS["efficientnet_b0"]["weights_path"]),
            )

    def load_model_version(
        self,
        architecture: str,
        version_tag: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        arch_key = normalize_architecture_key(architecture)
        spec = MODEL_SPECS[arch_key]
        resolved_weights, calib_path = self._resolve_paths_for_arch(arch_key, weights_path)

        if not resolved_weights.exists():
            error_msg = f"CRITICAL ERROR: Required {spec['display_name']} checkpoint not found at {resolved_weights}."
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolved_weights, map_location=device)
        ckpt_arch = normalize_architecture_key(checkpoint.get("architecture", arch_key))
        if ckpt_arch != arch_key:
            raise ValueError(
                f"Architecture mismatch in checkpoint {resolved_weights}: expected '{arch_key}', got '{ckpt_arch}'."
            )

        loaded_model = get_model(architecture=arch_key, num_classes=len(DIAGNOSTIC_CLASSES), pretrained=False)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.to(device)
        loaded_model.eval()

        loaded_calibrator = None
        if calib_path.exists():
            try:
                loaded_calibrator = TemperatureScaling.load(calib_path)
                logger.info(f"Loaded confidence calibrator from {calib_path} (T={loaded_calibrator.temperature:.4f})")
            except Exception as exc:
                logger.warning(f"Could not load calibration artifact {calib_path}: {exc}")

        with self._lock:
            self.device = device
            self.model = loaded_model
            self.calibrator = loaded_calibrator
            self.architecture_key = arch_key
            self.architecture = spec["display_name"]
            self.model_version_tag = version_tag or spec["default_version_tag"]
            self.active_weights_path = resolved_weights

        logger.info(
            f"Loaded {self.architecture} ({self.model_version_tag}) from {resolved_weights} "
            f"(epoch={checkpoint.get('epoch')}, val_bal_acc={checkpoint.get('val_balanced_acc', 0):.4f})"
        )

    def ensure_active_model(self, active_model: Optional[ModelVersion]):
        if active_model is None:
            return
        target_arch = normalize_architecture_key(active_model.architecture)
        if (
            self.model is None
            or self.architecture_key != target_arch
            or self.model_version_tag != active_model.version_tag
        ):
            self.load_model_version(
                architecture=active_model.architecture,
                version_tag=active_model.version_tag,
                weights_path=active_model.weights_path,
            )

    def predict_image(
        self,
        image_path: Path,
        generate_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference pipeline:
        1. Image Quality Assessment
        2. Preprocessing & Normalization (224x224, ImageNet)
        3. Lesion Region Localization & ROI Mask
        4. Forward Pass & Probability Softmax on active deployed model
        5. Post-hoc Temperature Scaling Confidence Calibration
        6. Top-3 and Top-N class ranking
        7. Grad-CAM Explainability Generation (heatmap & overlay)
        8. Clinical Risk Stratification
        """
        if self.model is None:
            self.initialize()

        inference_timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Quality Check
        quality_res = ImageQualityService.evaluate_image(image_path)
        quality_tier = quality_res["quality_tier"]

        # 2. Preprocess
        rgb_image, input_tensor = ImageProcessingService.load_and_preprocess(image_path)
        input_tensor = input_tensor.to(self.device)

        # Save authentic preprocessed 224x224 RGB image for inspection & reporting
        preprocessed_filename = f"{image_path.stem}_preprocessed.png"
        preprocessed_path = settings.UPLOAD_DIR / preprocessed_filename
        preprocessed_url = None
        try:
            from ml.src.preprocessing import unnormalize_tensor
            preproc_img = unnormalize_tensor(input_tensor)
            preproc_img.save(preprocessed_path, format="PNG")
            preprocessed_url = f"/api/v1/cases/files/{preprocessed_filename}"
        except Exception as e:
            logger.warning(f"Failed to save preprocessed image: {e}")

        # 3. Lesion Localization ROI Mask
        mask_filename = f"{image_path.stem}_roi_mask.png"
        mask_out_path = settings.UPLOAD_DIR / mask_filename
        roi_res = LesionSegmentationService.generate_roi_and_mask(image_path, mask_out_path)

        # 4. Model Forward Pass & Calibration under active lock
        with self._lock:
            active_model = self.model
            active_calibrator = self.calibrator
            active_arch = self.architecture
            active_version_tag = self.model_version_tag

            with torch.no_grad():
                logits = active_model(input_tensor)
                raw_probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

                if active_calibrator is not None:
                    cal_probabilities = active_calibrator.predict_proba(logits)[0]
                else:
                    cal_probabilities = raw_probabilities.copy()

            pred_idx = int(raw_probabilities.argmax())
            predicted_class = DIAGNOSTIC_CLASSES[pred_idx]
            raw_model_score = float(raw_probabilities[pred_idx])
            calibrated_confidence = float(cal_probabilities[pred_idx])

            gradcam_overlay_url = None
            gradcam_overlay_path = None
            gradcam_heatmap_url = None
            gradcam_heatmap_path = None

            if generate_gradcam and hasattr(active_model, "get_target_layer_for_gradcam"):
                try:
                    target_layer = active_model.get_target_layer_for_gradcam()
                    gradcam = GradCAM(active_model, target_layer)
                    tensor_grad = input_tensor.clone().requires_grad_(True)
                    _, heatmap_img, overlay_img = gradcam.generate_bundle(
                        original_image=rgb_image,
                        input_tensor=tensor_grad,
                        target_class_idx=pred_idx,
                        alpha=0.5,
                    )
                    gradcam.remove_hooks()

                    hm_filename = f"{image_path.stem}_gradcam_heatmap.png"
                    hm_path = settings.UPLOAD_DIR / hm_filename
                    heatmap_img.save(hm_path, format="PNG")
                    gradcam_heatmap_path = str(hm_path)
                    gradcam_heatmap_url = f"/api/v1/cases/files/{hm_filename}"

                    overlay_filename = f"{image_path.stem}_gradcam_overlay.png"
                    overlay_path = settings.UPLOAD_DIR / overlay_filename
                    overlay_img.save(overlay_path, format="PNG")
                    gradcam_overlay_path = str(overlay_path)
                    gradcam_overlay_url = f"/api/v1/cases/files/{overlay_filename}"
                except Exception as exc:
                    logger.warning(f"Grad-CAM generation failed: {exc}")

        # Top-N distribution sorted descending
        top_n_items = []
        top_3_items = []
        sorted_indices = raw_probabilities.argsort()[::-1]

        for rank, idx in enumerate(sorted_indices):
            cls_code = DIAGNOSTIC_CLASSES[idx]
            raw_p = float(raw_probabilities[idx])
            cal_p = float(cal_probabilities[idx])

            item = TopNItem(
                class_code=cls_code,
                class_name=CLASS_NAMES.get(cls_code, cls_code),
                probability=round(raw_p, 4),
                percentage_str=f"{raw_p * 100:.1f}%",
                raw_score=round(raw_p, 4),
                calibrated_score=round(cal_p, 4),
            )
            top_n_items.append(item)

            if rank < 3:
                top_3_items.append(
                    Top3PredictionItem(
                        class_code=cls_code,
                        class_name=CLASS_NAMES.get(cls_code, cls_code),
                        raw_score=round(raw_p, 4),
                        calibrated_score=round(cal_p, 4),
                        raw_percentage_str=f"{raw_p * 100:.1f}%",
                        calibrated_percentage_str=f"{cal_p * 100:.1f}%",
                    )
                )

        # 6. Risk Stratification (uses calibrated confidence when available)
        risk_tier, risk_rationale = RiskStratificationService.evaluate_risk(
            predicted_class=predicted_class,
            confidence=calibrated_confidence,
            quality_tier=quality_tier,
        )

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, predicted_class),
            "raw_model_score": round(raw_model_score, 4),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "confidence_score": round(raw_model_score, 4),
            "confidence_percentage_str": f"{raw_model_score * 100:.1f}%",
            "top_3_predictions": top_3_items,
            "top_n_probabilities": top_n_items,
            "risk_tier": risk_tier,
            "risk_rationale": risk_rationale,
            "preprocessed_image_path": str(preprocessed_path) if preprocessed_path.exists() else None,
            "preprocessed_image_url": preprocessed_url,
            "gradcam_overlay_path": gradcam_overlay_path,
            "gradcam_heatmap_path": gradcam_heatmap_path,
            "gradcam_url": gradcam_overlay_url,
            "gradcam_overlay_url": gradcam_overlay_url,
            "gradcam_heatmap_url": gradcam_heatmap_url,
            "roi_mask_path": roi_res["roi_mask_path"],
            "roi_mask_url": f"/api/v1/cases/files/{mask_filename}",
            "roi_bbox": roi_res["bbox"],
            "roi_area_ratio": roi_res["roi_area_ratio"],
            "quality_results": quality_res,
            "model_version": active_version_tag,
            "architecture": active_arch,
            "inference_timestamp": inference_timestamp,
            "inference_status": "SUCCESS",
        }
