from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for deep convolutional neural networks.
    Visualizes which spatial regions contributed most strongly to the predicted class score.
    Target layer for EfficientNet-B0: model.backbone.features[-1] (final Conv2dNormActivation block).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        # Backward pass with respect to target class logit
        target_score = output[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check hooks and target layer.")

        # Global average pooling of gradients: [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Linear combination: [1, 1, H, W]
        weighted_act = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only features that have a positive influence on the target class
        cam = torch.relu(weighted_act).squeeze().cpu().numpy()

        # Min-max normalization between 0 and 1
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def create_heatmap_image(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Resizes normalized heatmap to original image dimensions and applies colormap.
        Returns a colorized heatmap PIL Image in RGB format.
        """
        w, h = original_image.size
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        uint8_cam = np.uint8(255 * np.clip(resized_cam, 0.0, 1.0))
        color_cam_bgr = cv2.applyColorMap(uint8_cam, colormap)
        color_cam_rgb = cv2.cvtColor(color_cam_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(color_cam_rgb)

    def create_overlay(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Resizes heatmap to original image dimensions, applies colormap, and blends with original.
        Returns alpha-blended PIL Image in RGB format.
        """
        w, h = original_image.size
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        uint8_cam = np.uint8(255 * np.clip(resized_cam, 0.0, 1.0))

        color_cam_bgr = cv2.applyColorMap(uint8_cam, colormap)
        color_cam_rgb = cv2.cvtColor(color_cam_bgr, cv2.COLOR_BGR2RGB)

        orig_np = np.asarray(original_image.convert("RGB"))
        blended = np.uint8(orig_np * (1.0 - alpha) + color_cam_rgb * alpha)

        return Image.fromarray(blended)

    def generate_bundle(
        self,
        original_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, Image.Image, Image.Image]:
        """
        Generates:
        1. raw heatmap: np.ndarray [H_feat, W_feat] normalized to [0.0, 1.0]
        2. standalone heatmap image: PIL Image [H_orig, W_orig, 3] in RGB
        3. overlay image: PIL Image [H_orig, W_orig, 3] in RGB
        """
        heatmap = self.generate_heatmap(input_tensor, target_class_idx=target_class_idx)
        heatmap_img = self.create_heatmap_image(original_image, heatmap)
        overlay_img = self.create_overlay(original_image, heatmap, alpha=alpha)
        return heatmap, heatmap_img, overlay_img

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        self.remove_hooks()
