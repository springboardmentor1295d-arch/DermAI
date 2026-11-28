from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2


class BaseLesionSegmentor(ABC):
    """
    Abstract interface for lesion segmentation models.
    NOTE: PAD-UFES-20 does NOT contain ground-truth pixel segmentation masks.
    This interface serves as an extension point for future integration of supervised
    segmentation architectures (e.g., U-Net, Attention U-Net, DeepLabV3+) trained on
    mask-annotated datasets like ISIC.
    """
    @abstractmethod
    def segment(self, image_np: np.ndarray) -> np.ndarray:
        """Returns binary mask uint8 [0, 255] of lesion region."""
        pass


class ClassicalLesionLocalizer(BaseLesionSegmentor):
    """
    Unsupervised lesion region localization using color space conversion,
    adaptive Otsu thresholding, and morphological filtering.
    """
    def segment(self, image_np: np.ndarray) -> np.ndarray:
        # Resize to standard analysis resolution
        h, w = image_np.shape[:2]
        small = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Convert to CIE-Lab color space (L=luminance, a/b=chrominance)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Lesions typically show strong contrast in the blue-yellow (b) or luminance channel
        blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
        
        # Otsu thresholding with inverse
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological opening and closing to remove noise and bridge holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Resize mask back to original image size
        full_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return full_mask

    def localize_roi(self, image_np: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts binary mask and bounding box coordinates [x, y, w, h] of largest lesion contour.
        """
        mask = self.segment(image_np)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image_np.shape[:2]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            roi_area_ratio = round((bw * bh) / (w * h), 3)
        else:
            bx, by, bw, bh = 0, 0, w, h
            roi_area_ratio = 1.0

        roi_data = {
            "bbox": [bx, by, bw, bh],
            "roi_area_ratio": roi_area_ratio,
            "has_detected_lesion": len(contours) > 0
        }
        return mask, roi_data


class LesionSegmentationService:
    localizer = ClassicalLesionLocalizer()

    @classmethod
    def generate_roi_and_mask(cls, image_path: Path, output_mask_path: Path) -> Dict[str, Any]:
        with Image.open(image_path) as raw:
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
            np_img = np.asarray(rgb)

        mask, roi_info = cls.localizer.localize_roi(np_img)
        
        # Save visualization mask as PNG
        mask_pil = Image.fromarray(mask)
        mask_pil.save(output_mask_path, format="PNG")
        
        return {
            "roi_mask_path": str(output_mask_path),
            "bbox": roi_info["bbox"],
            "roi_area_ratio": roi_info["roi_area_ratio"],
            "has_detected_lesion": roi_info["has_detected_lesion"],
        }
