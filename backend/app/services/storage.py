import os
import uuid

import cv2
import numpy as np

from ..config import settings


def save_rgb(rgb: np.ndarray, subdir: str, name: str | None = None) -> str:
    """Save RGB array as JPEG under storage_dir/subdir; returns RELATIVE path (swap for S3/MinIO here)."""
    rel = os.path.join(subdir, (name or uuid.uuid4().hex) + ".jpg")
    full = os.path.join(settings.storage_dir, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    cv2.imwrite(full, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return rel.replace("\\", "/")


def load_rgb(rel: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(os.path.join(settings.storage_dir, rel)), cv2.COLOR_BGR2RGB)
