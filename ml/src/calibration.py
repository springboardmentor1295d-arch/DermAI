from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TemperatureScaling:
    """
    Temperature Scaling for post-hoc confidence calibration of deep neural networks.
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    
    A single scalar parameter T > 0 scales the model logits before softmax:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    
    Parameters are optimized strictly on validation logits to minimize NLL loss.
    Test logits are strictly held-out and evaluated without modification.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fits temperature T strictly on validation logits and labels.
        Does NOT use test set labels or data.
        """
        if not isinstance(val_logits, torch.Tensor):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels, dtype=torch.long)

        # Temperature parameter (log_temp ensures T stays strictly positive)
        log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            scaled_logits = val_logits / temp
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        fitted_temp = float(torch.exp(log_temp).item())
        # Constrain temperature to reasonable bounds [0.1, 10.0]
        self.temperature = max(0.1, min(10.0, fitted_temp))
        return self.temperature

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        return logits / self.temperature

    def predict_proba(self, logits: torch.Tensor) -> np.ndarray:
        """Calculates calibrated softmax probabilities."""
        scaled = self.calibrate_logits(logits)
        probs = torch.softmax(scaled, dim=-1)
        return probs.detach().cpu().numpy()

    def save(self, filepath: Path, metadata: Optional[Dict[str, Any]] = None):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": "temperature_scaling",
            "temperature": round(self.temperature, 5),
            "fitted_split": "validation",
            "metadata": metadata or {}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "TemperatureScaling":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration artifact not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(temperature=data.get("temperature", 1.0))


def compute_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log-Likelihood (cross-entropy loss)."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    return float(criterion(logits_t, labels_t).item())


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 6) -> float:
    """
    Computes multi-class Brier score:
    Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) with equal-width binning.
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / len(labels)) * bin_error

            bin_details.append({
                "bin_index": i + 1,
                "bin_range": [round(float(bin_lower), 3), round(float(bin_upper), 3)],
                "count": bin_count,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(bin_error, 4)
            })
        else:
            bin_details.append({
                "bin_index": i + 1,
                "bin_range": [round(float(bin_lower), 3), round(float(bin_upper), 3)],
                "count": 0,
                "confidence": 0.0,
                "accuracy": 0.0,
                "gap": 0.0
            })

    return float(ece), bin_details


def compute_accuracy_metrics(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Computes overall accuracy and balanced accuracy."""
    acc = float(np.mean(preds == labels))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    return acc, bal_acc


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    n_bins: int = 10,
    ece: Optional[float] = None
):
    """
    Renders an empirical reliability diagram with:
    - Diagonal dashed reference line for perfect calibration (y = x)
    - Empirical confidence vs accuracy bars
    - Calibration gap shading
    - ECE score annotation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ece is None:
        ece_val, bin_details = compute_ece(probs, labels, n_bins=n_bins)
    else:
        ece_val = ece
        _, bin_details = compute_ece(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Perfect Calibration")

    bin_midpoints = []
    accuracies = []
    gaps = []
    counts = []

    for b in bin_details:
        mid = (b["bin_range"][0] + b["bin_range"][1]) / 2.0
        bin_midpoints.append(mid)
        accuracies.append(b["accuracy"])
        gaps.append(b["gap"])
        counts.append(b["count"])

    bin_width = 1.0 / n_bins

    # Plot empirical accuracy bars
    ax.bar(
        bin_midpoints,
        accuracies,
        width=bin_width * 0.85,
        color="#2563eb",
        alpha=0.75,
        edgecolor="#1d4ed8",
        label="Outputs (Empirical Accuracy)"
    )

    # Plot calibration gap (difference between confidence and accuracy)
    confidences = [b["confidence"] for b in bin_details]
    ax.bar(
        bin_midpoints,
        gaps,
        bottom=accuracies,
        width=bin_width * 0.85,
        color="#f87171",
        alpha=0.45,
        edgecolor="#ef4444",
        linestyle="--",
        label="Calibration Gap"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=11, fontweight="medium")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="medium")
    ax.set_title(f"{title}\n(ECE = {ece_val * 100:.2f}%)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

