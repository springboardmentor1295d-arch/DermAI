import cv2
import numpy as np

from backend.app.services import preprocess, quality, segmentation


def synthetic(blur=False, dark=False):
    img = np.full((600, 800, 3), (200, 160, 140), np.uint8)
    cv2.circle(img, (400, 300), 110, (90, 50, 40), -1)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), 25)
    if dark:
        img = (img * 0.15).astype(np.uint8)
    return img


def test_good_quality():
    assert quality.assess(synthetic())["label"] in ("Good", "Acceptable")


def test_blur_detected():
    q = quality.assess(synthetic(blur=True))
    assert q["label"] == "Poor" and any("blur" in i.lower() for i in q["issues"])


def test_dark_detected():
    assert quality.assess(synthetic(dark=True))["label"] == "Poor"


def test_low_resolution():
    assert quality.assess(cv2.resize(synthetic(), (150, 110)))["label"] == "Poor"


def test_segmentation_finds_lesion():
    mask, bbox, ratio, method = segmentation.segment_lesion(synthetic())
    assert 0.02 < ratio < 0.6 and mask.shape == (600, 800)
    x, y, w, h = bbox
    assert x < 400 < x + w and y < 300 < y + h


def test_preprocess_shapes():
    img = synthetic(); out = preprocess.preprocess(img, (100, 100, 300, 300))
    assert out.shape == (300, 300, 3) and out.dtype == np.uint8
