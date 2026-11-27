"""Explainability: Grad-CAM++ for CNNs, gradient saliency fallback for Transformers."""
import cv2
import numpy as np
import torch


def _last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def _overlay(rgb, heat):
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    col = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.55, col, 0.45, 0)


def explain(model, x, class_idx, rgb_roi, arch: str):
    transformer = any(k in arch for k in ("vit", "swin", "deit"))
    if not transformer:
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            with GradCAMPlusPlus(model=model, target_layers=[_last_conv(model)]) as cam:
                g = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
            return _overlay(rgb_roi, g), "Grad-CAM++"
        except Exception:
            pass
    xx = x.clone().requires_grad_(True)
    model.zero_grad(); model(xx)[0, class_idx].backward()
    sal = xx.grad.abs().max(1)[0][0].numpy()
    sal = cv2.GaussianBlur(sal, (0, 0), 3)
    return _overlay(rgb_roi, sal), "Saliency"
