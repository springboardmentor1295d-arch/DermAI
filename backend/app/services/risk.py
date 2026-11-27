"""Risk categories are derived ONLY from the dataset's class labels (PAD-UFES-20 diagnostic groups):
malignant = BCC, SCC, MEL; pre-malignant = ACK; benign = NEV, SEK. No invented clinical thresholds.
The low-confidence / poor-quality rule is a project-defined research protocol (configurable)."""
from ..config import settings

CLASS_NAMES = {"ACK": "Actinic keratosis", "BCC": "Basal cell carcinoma", "MEL": "Melanoma",
               "NEV": "Melanocytic nevus", "SCC": "Squamous cell carcinoma", "SEK": "Seborrheic keratosis"}
CLASS_RISK = {"MEL": "High Risk", "BCC": "High Risk", "SCC": "High Risk", "ACK": "Moderate Risk",
              "NEV": "Low Risk", "SEK": "Low Risk"}


def assess(pred: str, confidence: float, quality_label: str):
    needs_review = confidence < settings.low_confidence_threshold or quality_label == "Poor"
    risk = "Requires Clinical Review" if needs_review else CLASS_RISK.get(pred, "Requires Clinical Review")
    return risk, needs_review
