"""PAD-UFES-20 dataset utilities (smartphone skin lesion images + clinical metadata)."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
NUM_COLS = ["age", "fitspatrick", "diameter_1", "diameter_2"]
CAT_COLS = ["gender", "region", "smoke", "drink", "skin_cancer_history",
            "cancer_history", "itch", "grew", "hurt", "changed", "bleed", "elevation"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_tf(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.15, 0.15, 0.1, 0.01),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])


def eval_tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def make_meta_encoder(train_df: pd.DataFrame):
    """Tabular encoder for clinical metadata (fit on TRAIN only to avoid leakage)."""
    enc = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNK")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
    ])
    enc.fit(_prep_meta(train_df))
    return enc


def _prep_meta(df):
    m = df[NUM_COLS + CAT_COLS].copy()
    for c in NUM_COLS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in CAT_COLS:
        m[c] = m[c].astype(str).replace({"nan": np.nan})
    return m


class PadDataset(Dataset):
    def __init__(self, df, transform, meta_encoder=None):
        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.labels = self.df["diagnostic"].map({c: i for i, c in enumerate(CLASSES)}).values
        self.meta = None
        if meta_encoder is not None:
            self.meta = meta_encoder.transform(_prep_meta(self.df)).astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.loc[i, "path"]).convert("RGB")
        x = self.tf(img)
        m = torch.from_numpy(self.meta[i]) if self.meta is not None else torch.zeros(1)
        return x, m, int(self.labels[i])
