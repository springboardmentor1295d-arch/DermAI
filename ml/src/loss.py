from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
import torch.nn as nn
from ml.src.dataset import DIAGNOSTIC_CLASSES, CLASS_TO_IDX


def compute_class_weights(split_csv_path: Path, split: str = "train") -> torch.Tensor:
    """
    Computes inverse frequency class weights:
    W_c = N / (C * N_c)
    Ensures rare classes like MEL (2.26%) receive higher gradient signal.
    """
    df = pd.read_csv(split_csv_path)
    train_df = df[df["split"].str.lower() == split.lower()]
    
    total_samples = len(train_df)
    num_classes = len(DIAGNOSTIC_CLASSES)
    
    counts = train_df["diagnostic"].str.upper().value_counts().to_dict()
    
    weights = []
    for cls in DIAGNOSTIC_CLASSES:
        count = counts.get(cls, 1)
        w = total_samples / (num_classes * count)
        weights.append(w)
        
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean is 1.0
    weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor


def get_loss_function(split_csv_path: Path, device: torch.device) -> nn.Module:
    weights = compute_class_weights(split_csv_path, split="train").to(device)
    return nn.CrossEntropyLoss(weight=weights)
