from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from ml.src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_INPUT_SIZE,
    get_inference_transforms,
    get_training_transforms,
    load_image_safely,
    preprocess_image,
    apply_dull_razor,
    unnormalize_tensor
)


class ImageProcessingService:
    """
    Image processing service integrating the centralized ml.src.preprocessing module.
    Maintains full backward compatibility with existing backend contracts.
    """
    IMAGENET_MEAN = IMAGENET_MEAN
    IMAGENET_STD = IMAGENET_STD
    TARGET_SIZE: Tuple[int, int] = DEFAULT_INPUT_SIZE

    @classmethod
    def get_inference_transforms(cls) -> T.Compose:
        return get_inference_transforms(target_size=cls.TARGET_SIZE)

    @classmethod
    def get_training_transforms(cls) -> T.Compose:
        return get_training_transforms(target_size=cls.TARGET_SIZE)

    @classmethod
    def load_and_preprocess(cls, image_path: Path) -> Tuple[Image.Image, torch.Tensor]:
        """
        Loads an image, applies EXIF orientation transpose, converts to RGB,
        and produces the standardized normalized tensor [1, 3, 224, 224].
        """
        return preprocess_image(image_path, target_size=cls.TARGET_SIZE, apply_hair_removal=False)

    @classmethod
    def apply_dull_razor(cls, image_np: np.ndarray) -> np.ndarray:
        """
        DullRazor algorithm for dark hair artifact attenuation.
        Uses morphological black hat to isolate thin dark structures and inpaints them.
        """
        return apply_dull_razor(image_np)
