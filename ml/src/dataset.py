from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DIAGNOSTIC_CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(DIAGNOSTIC_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(DIAGNOSTIC_CLASSES)}

CLASS_NAMES = {
    "BCC": "Basal Cell Carcinoma",
    "ACK": "Actinic Keratosis",
    "NEV": "Melanocytic Nevus",
    "SEK": "Seborrheic Keratosis",
    "SCC": "Squamous Cell Carcinoma",
    "MEL": "Melanoma"
}

from ml.src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_training_transforms,
    get_inference_transforms,
    load_image_safely,
)


def get_transforms(split: str = "train") -> T.Compose:
    if split.lower() == "train":
        return get_training_transforms()
    else:
        return get_inference_transforms()


class PadUfesDataset(Dataset):
    """
    Dataset loader for PAD-UFES-20 consuming the fixed patient-level stratified split.
    Guarantees no patient overlap across splits.
    """
    def __init__(
        self,
        split_csv_path: Path = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv"),
        images_dir: Path = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images"),
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.split_csv_path = Path(split_csv_path)
        self.images_dir = Path(images_dir)
        self.split = split.lower()
        self.transform = transform if transform is not None else get_transforms(self.split)

        if not self.split_csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv_path}")

        df = pd.read_csv(self.split_csv_path)
        self.df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        
        # Build image lookup index across images directory recursively
        self.image_lookup: Dict[str, Path] = {}
        for img_file in self.images_dir.rglob("*.png"):
            if img_file.is_file():
                self.image_lookup[img_file.name.lower()] = img_file

        # Verify all records have corresponding image files
        missing = []
        for img_id in self.df["img_id"]:
            if str(img_id).strip().lower() not in self.image_lookup:
                missing.append(img_id)

        if missing:
            raise FileNotFoundError(f"Found {len(missing)} missing images in split {self.split}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row["img_id"]).strip()
        img_path = self.image_lookup[img_id.lower()]
        diagnostic = str(row["diagnostic"]).strip().upper()
        label_idx = CLASS_TO_IDX[diagnostic]

        image = load_image_safely(img_path)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        return image_tensor, label_idx, img_id
