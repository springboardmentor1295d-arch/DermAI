"""Lesion detection/segmentation WITHOUT ground-truth masks (PAD-UFES-20 has none):
GrabCut initialised from a central rectangle + morphology + largest component.
Swap `segment_lesion` for a U-Net / SAM-based model later; the interface (mask, bbox, ratio) stays the same."""
import cv2
import numpy as np


def segment_lesion(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    s = 512 / max(h, w) if max(h, w) > 512 else 1.0
    small = cv2.resize(rgb, None, fx=s, fy=s) if s < 1 else rgb.copy()
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    method = "grabcut"
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, (int(.12 * sw), int(.12 * sh), int(.76 * sw), int(.76 * sh)),
                    np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
        if n > 1:  # largest component, preferring ones near the centre
            score = [stats[i, cv2.CC_STAT_AREA] / (1 + np.hypot(cen[i][0] - sw / 2, cen[i][1] - sh / 2) / sw) for i in range(1, n)]
            m = np.where(lab == 1 + int(np.argmax(score)), 255, 0).astype(np.uint8)
        ratio = m.mean() / 255
    except cv2.error:
        ratio = 0
    if not (0.005 < ratio < 0.90):  # degenerate -> central ellipse fallback
        method = "fallback_ellipse"
        m = np.zeros((sh, sw), np.uint8)
        cv2.ellipse(m, (sw // 2, sh // 2), (int(.3 * sw), int(.3 * sh)), 0, 0, 360, 255, -1)
    full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    x, y, bw, bh = cv2.boundingRect(full)
    pad = int(0.15 * max(bw, bh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return full, (x0, y0, x1 - x0, y1 - y0), float(full.mean() / 255), method


def overlay_mask(rgb, mask):
    out = rgb.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tint = np.zeros_like(rgb); tint[mask > 0] = (0, 255, 120)
    out = cv2.addWeighted(out, 1.0, tint, 0.25, 0)
    cv2.drawContours(out, cnts, -1, (0, 255, 120), max(2, rgb.shape[1] // 250))
    return out
