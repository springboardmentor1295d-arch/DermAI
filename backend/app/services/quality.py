"""Image quality assessment (classical CV, no DL). Thresholds are starting points: tune on your own phone images."""
import cv2
import numpy as np

T = dict(blur_poor=15, blur_ok=40, dark_poor=45, dark_ok=70, bright_poor=220, bright_ok=200,
         contrast_poor=18, contrast_ok=30, res_poor=224, res_ok=400)


def assess(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = 640 / max(h, w)  # normalise scale so blur score is comparable across phone resolutions
    g = cv2.resize(gray, None, fx=s, fy=s) if s < 1 else gray
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright, contrast = float(gray.mean()), float(gray.std())
    sat_clip = float((gray >= 250).mean())
    issues, level = [], 0

    def flag(cond_poor, cond_ok, msg):
        nonlocal level
        if cond_poor: issues.append(("poor", msg)); level = max(level, 2)
        elif cond_ok: issues.append(("acceptable", msg)); level = max(level, 1)

    flag(blur < T["blur_poor"], blur < T["blur_ok"], "Image is blurred / out of focus")
    flag(bright < T["dark_poor"], bright < T["dark_ok"], "Image is too dark")
    flag(bright > T["bright_poor"] or sat_clip > 0.35, bright > T["bright_ok"] or sat_clip > 0.15, "Image is overexposed")
    flag(contrast < T["contrast_poor"], contrast < T["contrast_ok"], "Low contrast")
    flag(min(h, w) < T["res_poor"], min(h, w) < T["res_ok"], "Resolution too low")
    return dict(label=["Good", "Acceptable", "Poor"][level], blur_score=round(blur, 1), brightness=round(bright, 1),
                contrast=round(contrast, 1), width=w, height=h, overexposed_fraction=round(sat_clip, 3),
                issues=[m for _, m in issues])
