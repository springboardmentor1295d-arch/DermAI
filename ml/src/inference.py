from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_NAMES
from ml.src.models import get_model, EfficientNetB0Transfer
from ml.src.preprocessing import load_image_safely, preprocess_image, DEFAULT_INPUT_SIZE
from ml.src.calibration import TemperatureScaling

MODEL_VERSION = "v1.0.0-efficientnet_b0"
DEFAULT_CHECKPOINT_PATH = Path("artifacts/ml/efficientnet_b0_best.pth")
DEFAULT_CALIBRATION_PATH = Path("artifacts/ml/temperature_scaling.json")


class EfficientNetB0InferenceEngine:
    """
    Reusable, production-grade inference engine for the deployed EfficientNet-B0 classifier.
    Adheres strictly to the Critical Model Selection Rule:
    - EfficientNet-B0 is the sole deployed classification model.
    - No MobileNet fallback.
    - Strict validation of input images and model weights.
    - Top-3 ranking with raw and calibrated confidence scores.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        calibration_path: Optional[Union[str, Path]] = DEFAULT_CALIBRATION_PATH,
        device: Optional[torch.device] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[EfficientNetB0Transfer] = None
        self.calibrator: Optional[TemperatureScaling] = None
        self.classes: List[str] = DIAGNOSTIC_CLASSES
        self.class_names: Dict[str, str] = CLASS_NAMES
        self.model_version: str = MODEL_VERSION
        self.is_loaded: bool = False

        self._load_model()
        self._load_calibration()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Critical Model Error: Required EfficientNet-B0 checkpoint not found at {self.checkpoint_path}"
            )

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            arch = ckpt.get("architecture", "efficientnet_b0")
            if arch != "efficientnet_b0":
                raise ValueError(
                    f"Checkpoint architecture mismatch: expected 'efficientnet_b0', got '{arch}'"
                )

            model = get_model(architecture="efficientnet_b0", num_classes=len(self.classes), pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.model = model
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Failed to load EfficientNet-B0 checkpoint from {self.checkpoint_path}: {e}")

    def _load_calibration(self):
        if self.calibration_path and self.calibration_path.exists():
            try:
                self.calibrator = TemperatureScaling.load(self.calibration_path)
            except Exception:
                self.calibrator = None
        else:
            self.calibrator = None

    def validate_image_source(self, image_source: Union[str, Path, bytes, Image.Image]) -> Tuple[Image.Image, int, int, str]:
        """
        Validates image source format, size, readability, and dimensions.
        Returns: (PIL RGB Image, width, height, format_string)
        """
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Image file does not exist: {p}")
            valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
            if p.suffix.lower() not in valid_exts:
                raise ValueError(f"Unsupported image file extension: {p.suffix}. Allowed: {valid_exts}")

        rgb_image = load_image_safely(image_source)
        w, h = rgb_image.size
        if w < 32 or h < 32:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small for clinical inference. Minimum: 32x32.")

        return rgb_image, w, h, rgb_image.format or "RGB"

    def predict(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference using EfficientNet-B0:
        1. Validates and loads image
        2. Applies standard ImageNet preprocessing
        3. Computes raw logits and softmax scores
        4. Applies post-hoc temperature scaling calibration
        5. Computes Top-k predictions with safe bounds
        6. Returns structured response with timestamps, version, and status
        """
        inference_timestamp = datetime.now(timezone.utc).isoformat()

        if not self.is_loaded or self.model is None:
            return {
                "inference_status": "MODEL_ERROR",
                "error_message": "EfficientNet-B0 model is not loaded.",
                "inference_timestamp": inference_timestamp,
                "model_version": self.model_version,
                "predicted_class": None,
                "raw_model_score": None,
                "calibrated_confidence": None,
                "top_3_predictions": []
            }

        try:
            rgb_image, width, height, img_fmt = self.validate_image_source(image_source)
        except Exception as e:
            return {
                "inference_status": "INVALID_IMAGE",
                "error_message": str(e),
                "inference_timestamp": inference_timestamp,
                "model_version": self.model_version,
                "predicted_class": None,
                "raw_model_score": None,
                "calibrated_confidence": None,
                "top_3_predictions": []
            }

        try:
            _, input_tensor = preprocess_image(rgb_image, target_size=DEFAULT_INPUT_SIZE)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                if self.calibrator is not None:
                    cal_probs = self.calibrator.predict_proba(logits)[0]
                else:
                    cal_probs = raw_probs.copy()

            pred_idx = int(np.argmax(raw_probs))
            predicted_class = self.classes[pred_idx]
            raw_model_score = float(raw_probs[pred_idx])
            calibrated_confidence = float(cal_probs[pred_idx])

            # Top-K ranking (safely handling fewer than K classes)
            k = min(top_k, len(self.classes))
            sorted_indices = np.argsort(raw_probs)[::-1][:k]

            top_predictions = []
            for idx in sorted_indices:
                cls_code = self.classes[idx]
                r_score = float(raw_probs[idx])
                c_score = float(cal_probs[idx])
                top_predictions.append({
                    "class_code": cls_code,
                    "class_name": self.class_names.get(cls_code, cls_code),
                    "raw_score": round(r_score, 4),
                    "calibrated_score": round(c_score, 4),
                    "raw_percentage_str": f"{r_score * 100:.1f}%",
                    "calibrated_percentage_str": f"{c_score * 100:.1f}%"
                })

            all_probabilities = {
                self.classes[i]: {
                    "raw_score": round(float(raw_probs[i]), 4),
                    "calibrated_score": round(float(cal_probs[i]), 4)
                }
                for i in range(len(self.classes))
            }

            return {
                "inference_status": "SUCCESS",
                "error_message": None,
                "predicted_class": predicted_class,
                "class_name": self.class_names.get(predicted_class, predicted_class),
                "raw_model_score": round(raw_model_score, 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
                "top_3_predictions": top_predictions,
                "all_probabilities": all_probabilities,
                "input_dimensions": [width, height],
                "image_channels": "RGB",
                "preprocessing": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=[224, 224])",
                "model_version": self.model_version,
                "checkpoint_path": str(self.checkpoint_path),
                "inference_timestamp": inference_timestamp
            }

        except Exception as e:
            return {
                "inference_status": "FAILED",
                "error_message": f"Inference execution failed: {str(e)}",
                "inference_timestamp": inference_timestamp,
                "model_version": self.model_version,
                "predicted_class": None,
                "raw_model_score": None,
                "calibrated_confidence": None,
                "top_3_predictions": []
            }

