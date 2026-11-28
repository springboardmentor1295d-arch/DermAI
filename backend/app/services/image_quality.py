from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image, ImageOps
import cv2
from backend.app.models.image import QualityTier
from ml.src.preprocessing import center_square_crop


class ImageQualityService:
    # Heuristic thresholds derived from PAD-UFES-20 EDA & mobile camera validation:
    # Evaluated on the centered square ROI matching model input geometry.
    BLUR_THRESHOLD: float = 25.0
    LOW_LIGHT_THRESHOLD: float = 55.0
    OVEREXPOSURE_THRESHOLD: float = 215.0
    GLARE_PIXEL_RATIO_THRESHOLD: float = 0.25
    MIN_CONTRAST_STD: float = 12.0
    MIN_RESOLUTION: int = 128

    @classmethod
    def evaluate_image(cls, image_path: Path) -> Dict[str, Any]:
        """
        Calculates heuristic image quality indicators on EXIF-corrected, center-cropped RGB input.
        Checks:
        - Image resolution (minimum 128x128)
        - Motion/focus blur (Laplacian variance)
        - Poor lighting / excessive darkness or brightness / flash glare
        - Low contrast / unclear lesion separation
        Note: Advisory indicators only; does not block valid clinical captures unnecessarily.
        """
        flags: List[str] = []

        with Image.open(image_path) as raw_img:
            exif_obj = raw_img.getexif()
            orientation_tag = exif_obj.get(274, 1) if exif_obj else 1
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            orig_width, orig_height = img.size

        # Evaluate quality on the central lesion crop (matching inference framing)
        roi_img = center_square_crop(img)
        roi_width, roi_height = roi_img.size

        # Check resolution
        resolution_ok = 1
        if roi_width < cls.MIN_RESOLUTION or roi_height < cls.MIN_RESOLUTION:
            flags.append("low_resolution")
            resolution_ok = 0

        # Standardize to 512x512 thumbnail for consistent scale-invariant blur/luminance measurement
        img_thumb = roi_img.copy()
        img_thumb.thumbnail((512, 512))
        np_img = np.asarray(img_thumb)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # 1. Blur indicator: Discrete Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < cls.BLUR_THRESHOLD:
            flags.append("blur_detected")

        # 2. Exposure / Glare indicator: Mean brightness & clipped highlights
        brightness_mean = float(gray.mean())
        glare_ratio = float(np.mean(gray > 245))
        if brightness_mean < cls.LOW_LIGHT_THRESHOLD:
            flags.append("underexposed_low_light")
        elif brightness_mean > cls.OVEREXPOSURE_THRESHOLD or glare_ratio > cls.GLARE_PIXEL_RATIO_THRESHOLD:
            flags.append("overexposed_or_glare")

        # 3. Contrast indicator: Standard deviation of pixel luminance
        contrast_std = float(gray.std())
        if contrast_std < cls.MIN_CONTRAST_STD:
            flags.append("low_contrast")

        # Determine composite quality tier
        if len(flags) == 0:
            quality_tier = QualityTier.GOOD
            advisory_message = "Image quality is good. Sharpness and illumination are suitable for analysis."
        elif len(flags) == 1 and "blur_detected" not in flags and "low_resolution" not in flags:
            quality_tier = QualityTier.ACCEPTABLE
            advisory_message = "Image quality is acceptable. Minor lighting or contrast variation detected; ensure affected skin area is clearly visible."
        else:
            quality_tier = QualityTier.POOR
            advisory_message = "Image quality is low. Please capture a clear, well-lit image with the affected skin area visible."

        return {
            "width": orig_width,
            "height": orig_height,
            "exif_orientation_corrected": orientation_tag != 1,
            "blur_score": round(laplacian_var, 2),
            "brightness_mean": round(brightness_mean, 2),
            "contrast_std": round(contrast_std, 2),
            "resolution_ok": resolution_ok,
            "quality_tier": quality_tier,
            "advisory_flags": flags,
            "advisory_message": advisory_message,
        }
