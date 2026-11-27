"""Model serving: loads one checkpoint, or several (ensemble: softmax average). Temperature-scaled confidence."""
import time

import numpy as np

_state = {"key": None, "models": [], "cfg": None}


def _load(paths):
    import torch
    from ml.models import build_model
    models, cfg = [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        if ck.get("use_meta"):
            raise RuntimeError("Metadata-fusion checkpoints are research-only; serve an image-only model.")
        m = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append((m, ck.get("temperature", 1.0), ck["arch"]))
        cfg = cfg or dict(classes=ck["classes"], img_size=ck["img_size"])
    return models, cfg


def get_models(path_string: str):
    if _state["key"] != path_string:
        paths = [p.strip() for p in path_string.split(",") if p.strip()]
        _state["models"], _state["cfg"] = _load(paths); _state["key"] = path_string
    return _state["models"], _state["cfg"]


def _tensor(rgb, size):
    import torch
    from torchvision import transforms as T
    tf = T.Compose([T.ToPILImage(), T.Resize((size, size)), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return tf(rgb).unsqueeze(0)


def predict(path_string: str, rgb_roi: np.ndarray):
    import torch
    models, cfg = get_models(path_string)
    x = _tensor(rgb_roi, cfg["img_size"])
    t0 = time.time()
    with torch.no_grad():
        probs = np.mean([torch.softmax(m(x) / T, 1)[0].numpy() for m, T, _ in models], axis=0)
    order = np.argsort(-probs)
    classes = cfg["classes"]
    return dict(probs={c: float(p) for c, p in zip(classes, probs)}, pred=classes[order[0]],
                confidence=float(probs[order[0]]), top3=[(classes[i], float(probs[i])) for i in order[:3]],
                pred_idx=int(order[0]), x=x, primary=models[0][0], arch=models[0][2],
                ms=int((time.time() - t0) * 1000))
