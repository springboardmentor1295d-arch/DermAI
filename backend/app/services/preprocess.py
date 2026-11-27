import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(data: bytes) -> np.ndarray:
    """Decode + EXIF orientation correction (phone photos are often rotated via EXIF)."""
    import io
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return np.array(im)


def remove_hair(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    _, m = cv2.threshold(bh, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(rgb, m, 1, cv2.INPAINT_TELEA)


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Colour normalisation across different phone cameras / lighting."""
    x = rgb.astype(np.float32)
    x *= x.mean() / (x.reshape(-1, 3).mean(0) + 1e-6)
    return np.clip(x, 0, 255).astype(np.uint8)


def enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop_roi(rgb, bbox):
    x, y, w, h = bbox
    return rgb[y:y + h, x:x + w]


def preprocess(rgb: np.ndarray, bbox=None) -> np.ndarray:
    """Raw -> (crop to ROI) -> colour normalisation. Resize/normalise tensors happen in the inference transform
    so that serving matches training (torchvision Resize + ImageNet Normalize)."""
    out = rgb if bbox is None else crop_roi(rgb, bbox)
    return gray_world(out)
