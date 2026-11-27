"""Train one architecture on PAD-UFES-20.
Example: python -m ml.train --arch efficientnet_b0 --loss focal --epochs 25
         python -m ml.train --arch convnext_tiny --use-meta        # image + clinical metadata fusion
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import CLASSES, PadDataset, eval_tf, make_meta_encoder, train_tf
from .evaluate import compute_metrics, expected_calibration_error, fit_temperature, plot_confusion
from .models import build_model


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma, self.weight = gamma, weight

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loaders(df, size, batch, workers, use_meta, sampler=True):
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    enc = make_meta_encoder(tr) if use_meta else None
    d_tr, d_va, d_te = (PadDataset(x, t, enc) for x, t in ((tr, train_tf(size)), (va, eval_tf(size)), (te, eval_tf(size))))
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=torch.cuda.is_available())
    if sampler:
        cnt = np.bincount(d_tr.labels, minlength=len(CLASSES)).clip(min=1)
        w = (1.0 / cnt)[d_tr.labels]
        l_tr = DataLoader(d_tr, sampler=WeightedRandomSampler(w, len(w), replacement=True), **kw)
    else:
        l_tr = DataLoader(d_tr, shuffle=True, **kw)
    return l_tr, DataLoader(d_va, **kw), DataLoader(d_te, **kw), d_tr, enc


@torch.no_grad()
def predict(model, loader, device, use_meta):
    model.eval(); L, Y = [], []
    for x, m, y in loader:
        x, m = x.to(device), m.to(device)
        L.append((model(x, m) if use_meta else model(x)).float().cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--splits", default="ml/splits.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="focal")
    ap.add_argument("--no-sampler", action="store_true")
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="ml/runs")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{a.arch}{'_meta' if a.use_meta else ''}"
    out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.splits)
    l_tr, l_va, l_te, d_tr, enc = make_loaders(df, a.img_size, a.batch, a.workers, a.use_meta, not a.no_sampler)
    meta_dim = d_tr.meta.shape[1] if a.use_meta else 0
    if enc is not None:
        joblib.dump(enc, out / "meta_encoder.joblib")
    model = build_model(a.arch, len(CLASSES), True, meta_dim).to(dev)

    cnt = np.bincount(d_tr.labels, minlength=len(CLASSES)).clip(min=1)
    cw = torch.tensor(len(d_tr) / (len(CLASSES) * cnt), dtype=torch.float32, device=dev)
    crit = {"ce": nn.CrossEntropyLoss(), "weighted_ce": nn.CrossEntropyLoss(weight=cw),
            "focal": FocalLoss(2.0, cw if a.no_sampler else None)}[a.loss]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    if a.mlflow:
        import mlflow
        mlflow.set_experiment("dermai"); mlflow.start_run(run_name=name); mlflow.log_params(vars(a))

    best, bad, hist = -1, 0, []
    for ep in range(a.epochs):
        model.train(); t0, tot = time.time(), 0.0
        for x, m, y in l_tr:
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = crit(model(x, m) if a.use_meta else model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(y)
        sched.step()
        lg, yv = predict(model, l_va, dev, a.use_meta)
        met = compute_metrics(yv, softmax(lg), CLASSES)
        rec = dict(epoch=ep, loss=tot / len(d_tr), val_macro_f1=met["macro_f1"],
                   val_bal_acc=met["balanced_accuracy"], sec=round(time.time() - t0, 1))
        hist.append(rec); print(rec)
        if a.mlflow:
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
        if met["macro_f1"] > best:
            best, bad = met["macro_f1"], 0
            torch.save(model.state_dict(), out / "_best_state.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping"); break

    model.load_state_dict(torch.load(out / "_best_state.pt", map_location=dev))
    lv, yv = predict(model, l_va, dev, a.use_meta)
    temp = fit_temperature(lv, yv)
    lt, yt = predict(model, l_te, dev, a.use_meta)
    probs = softmax(lt / temp)
    metrics = compute_metrics(yt, probs, CLASSES)
    metrics.update(temperature=temp, ece=expected_calibration_error(probs, yt), arch=a.arch,
                   use_meta=a.use_meta, history=hist, n_test=int(len(yt)))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    plot_confusion(metrics["confusion_matrix"], CLASSES, out / "confusion_matrix.png", name)
    torch.save(dict(arch=a.arch, classes=CLASSES, img_size=a.img_size, state_dict={k: v.cpu() for k, v in model.state_dict().items()}, temperature=temp, use_meta=a.use_meta,
        metrics={k: metrics[k] for k in ("macro_f1", "balanced_accuracy", "mcc", "accuracy", "macro_auc_ovr")}),
        out / "best.pt")
    (out / "_best_state.pt").unlink(missing_ok=True)
    print(f"TEST macro-F1={metrics['macro_f1']:.3f} bal-acc={metrics['balanced_accuracy']:.3f} "
          f"MEL recall={metrics['per_class']['MEL']['sensitivity']:.3f}")
    if a.mlflow:
        mlflow.log_metrics({"test_macro_f1": metrics["macro_f1"], "test_bal_acc": metrics["balanced_accuracy"]})
        mlflow.log_artifact(str(out / "metrics.json")); mlflow.end_run()


if __name__ == "__main__":
    main()
