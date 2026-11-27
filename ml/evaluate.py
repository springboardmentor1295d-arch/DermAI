"""Metrics: accuracy, macro-F1, balanced accuracy, MCC, sensitivity, specificity, OvR AUC, confusion matrix."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


def compute_metrics(y_true, probs, classes):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        try:
            auc = float(roc_auc_score((y_true == i).astype(int), probs[:, i]))
        except ValueError:
            auc = None
        per_class[c] = dict(precision=float(p[i]), sensitivity=float(r[i]), f1=float(f[i]),
                            specificity=float(tn / (tn + fp)) if (tn + fp) else None,
                            auc_ovr=auc, support=int(s[i]), TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn))
    aucs = [v["auc_ovr"] for v in per_class.values() if v["auc_ovr"] is not None]
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                mcc=float(matthews_corrcoef(y_true, y_pred)),
                macro_auc_ovr=float(np.mean(aucs)) if aucs else None,
                per_class=per_class, confusion_matrix=cm.tolist())


def plot_confusion(cm, classes, path, title="Confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def fit_temperature(logits, labels):
    """Temperature scaling on validation logits (calibrates confidence scores)."""
    logits = torch.as_tensor(logits, dtype=torch.float32); labels = torch.as_tensor(labels)
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / t.clamp(min=0.05), labels); loss.backward(); return loss
    opt.step(closure)
    return float(t.clamp(min=0.05).item())


def expected_calibration_error(probs, y_true, bins=10):
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, pred = probs.max(1), probs.argmax(1)
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return float(ece)
