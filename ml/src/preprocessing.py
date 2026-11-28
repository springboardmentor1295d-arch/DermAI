from pathlib import Path
from typing import Tuple, Union, Optional, List
import io
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
import torchvision.transforms as T

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)


def get_inference_transforms(target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> T.Compose:
    """
    Returns deterministic transforms for validation, test, and production inference.
    Guarantees no stochastic augmentation.
    """
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transforms(target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> T.Compose:
    """
    Returns training transforms incorporating stochastic clinical image augmentations:
    - Bilinear resize to target dimension
    - Random horizontal and vertical flips (dermoscopic orientation invariance)
    - Mild ColorJitter (brightness and contrast variation across cameras)
    - Standard ImageNet tensor normalization
    """
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_safely(image_source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """
    Safely loads an image, validates its format, applies EXIF orientation transpose,
    and converts to standard 3-channel RGB.
    
    Handles:
    - Path or string filepath
    - Raw bytes or byte stream
    - Existing PIL Image instances
    - Color modes: L (grayscale), RGBA (alpha channel), CMYK, P (palette), 1 (binary)
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, bytes):
        try:
            img = Image.open(io.BytesIO(image_source))
        except Exception as e:
            raise ValueError(f"Failed to decode image from bytes: {e}")
    elif isinstance(image_source, (str, Path)):
        img_path = Path(image_source)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")
        try:
            img = Image.open(img_path)
        except Exception as e:
            raise ValueError(f"Corrupted or unsupported image file at {img_path}: {e}")
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    # Handle EXIF orientation tags common in smartphone photos
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # If EXIF reading fails, continue safely

    # Convert safely to standard RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess_image(
    image_source: Union[str, Path, bytes, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    apply_hair_removal: bool = False
) -> Tuple[Image.Image, torch.Tensor]:
    """
    End-to-end preprocessing pipeline for inference:
    1. Safe image loading and EXIF orientation normalization
    2. Conversion to RGB
    3. Optional DullRazor hair attenuation
    4. Deterministic resize and tensor normalization
    
    Returns:
    - rgb_image: PIL Image in RGB format
    - tensor: torch.FloatTensor of shape [1, 3, H, W]
    """
    rgb_image = load_image_safely(image_source)

    if apply_hair_removal:
        np_img = np.asarray(rgb_image)
        cleaned_np = apply_dull_razor(np_img)
        rgb_image = Image.fromarray(cleaned_np)

    transform = get_inference_transforms(target_size=target_size)
    tensor = transform(rgb_image).unsqueeze(0)  # Shape: [1, 3, target_size[0], target_size[1]]

    # Defensive validation assertions
    if tensor.shape != (1, 3, target_size[0], target_size[1]):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}, expected {(1, 3, target_size[0], target_size[1])}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"Unexpected tensor dtype: {tensor.dtype}, expected torch.float32")

    return rgb_image, tensor


def apply_dull_razor(image_np: np.ndarray) -> np.ndarray:
    """
    DullRazor morphological hair attenuation algorithm.
    Detects thin dark linear hair structures and replaces them using Telea inpainting.
    """
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB numpy array")

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned


def unnormalize_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Inverts ImageNet normalization on a tensor [C, H, W] or [1, C, H, W]
    and converts back to a viewable RGB PIL Image for visual inspection and reporting.
    """
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    unnorm = t * std + mean
    unnorm = torch.clamp(unnorm, 0.0, 1.0)
    unnorm_np = (unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(unnorm_np)
