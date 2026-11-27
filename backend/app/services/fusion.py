from pathlib import Path

import time
import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image
from ml.models import build_model
from ml.dataset import CLASSES, eval_tf, _prep_meta


class FusionWrapper(torch.nn.Module):
    def __init__(self, model, metadata):
        super().__init__()
        self.model = model
        self.register_buffer("metadata", metadata)

    def forward(self, x):
        return self.model(
            x, self.metadata.expand(x.shape[0], -1)
        )


def predict(path, rgb, metadata):
    start = time.time()
    path = Path(path)

    ck = torch.load(
        path, map_location="cpu", weights_only=True
    )

    encoder = joblib.load(
        path.with_name(path.stem + "_meta_encoder.joblib")
    )

    if not ck.get("use_meta"):
        raise ValueError("Not a metadata fusion model")

    if list(ck["classes"]) != CLASSES:
        raise ValueError("Class order mismatch")

    fields = encoder.feature_names_in_.tolist()

    missing = [
        k for k in fields
        if k not in metadata or metadata[k] is None
        or str(metadata[k]).strip() == ""
    ]

    if missing:
        raise ValueError(
            "Missing clinical fields: " + ", ".join(missing)
        )

    numeric = fields[:4]

    for key in numeric:
        if not np.isfinite(float(metadata[key])):
            raise ValueError("Invalid numeric field: " + key)

    categories = encoder.named_transformers_[
        "cat"
    ].named_steps["oh"].categories_

    for key, allowed in zip(fields[4:], categories):
        if str(metadata[key]) not in allowed:
            raise ValueError("Invalid category: " + key)

    row = pd.DataFrame(
        [{k: metadata[k] for k in fields}]
    )

    encoded = encoder.transform(
        _prep_meta(row)
    ).astype("float32")

    meta = torch.from_numpy(encoded)

    model = build_model(
        ck["arch"],
        len(CLASSES),
        pretrained=False,
        meta_dim=meta.shape[1]
    )

    model.load_state_dict(ck["state_dict"])
    model.eval()

    wrapped = FusionWrapper(model, meta)
    wrapped.eval()

    x = eval_tf(ck["img_size"])(
        Image.fromarray(rgb).convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():
        logits = wrapped(x)
        probs = torch.softmax(
            logits / ck.get("temperature", 1.0),
            dim=1
        )[0].numpy()

    order = np.argsort(-probs)

    return dict(
        probs={
            c: float(p)
            for c, p in zip(CLASSES, probs)
        },
        pred=CLASSES[order[0]],
        confidence=float(probs[order[0]]),
        top3=[
            (CLASSES[i], float(probs[i]))
            for i in order[:3]
        ],
        pred_idx=int(order[0]),
        x=x,
        primary=wrapped,
        arch=ck["arch"],
        ms=int((time.time() - start) * 1000)
    )
