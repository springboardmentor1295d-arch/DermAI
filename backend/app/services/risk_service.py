from typing import Tuple, Dict, Any
from backend.app.models.prediction import RiskTier
from backend.app.models.image import QualityTier

DISEASE_METADATA = {
    "MEL": {
        "name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "baseline_risk": RiskTier.HIGH,
        "clinical_significance": "Most dangerous form of skin cancer, arising from melanocytes.",
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer (Non-Melanoma)",
        "baseline_risk": RiskTier.HIGH,
        "clinical_significance": "Most common skin cancer. Locally invasive; low metastatic potential.",
    },
    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "category": "Malignant Skin Cancer (Non-Melanoma)",
        "baseline_risk": RiskTier.HIGH,
        "clinical_significance": "Second most common skin cancer; potential to metastasize if untreated.",
    },
    "ACK": {
        "name": "Actinic Keratosis",
        "category": "Pre-Malignant Lesion",
        "baseline_risk": RiskTier.MODERATE,
        "clinical_significance": "Sun-induced pre-cancerous lesion that can progress to SCC.",
    },
    "NEV": {
        "name": "Melanocytic Nevus",
        "category": "Benign Pigmented Lesion (Mole)",
        "baseline_risk": RiskTier.LOW,
        "clinical_significance": "Common benign proliferation of melanocytes; monitor for ABCD changes.",
    },
    "SEK": {
        "name": "Seborrheic Keratosis",
        "category": "Benign Keratinocytic Lesion",
        "baseline_risk": RiskTier.LOW,
        "clinical_significance": "Non-cancerous skin growth common with aging; harmless.",
    },
}


class RiskStratificationService:
    @classmethod
    def evaluate_risk(
        cls,
        predicted_class: str,
        confidence: float,
        quality_tier: QualityTier
    ) -> Tuple[RiskTier, str]:
        """
        Determines the clinical risk category and rationale.
        Rules:
        - If quality is POOR or confidence < 0.50: Flag as REQUIRES_REVIEW.
        - Otherwise, stratify based on documented clinical pathology (Malignant -> HIGH, Pre-malignant -> MODERATE, Benign -> LOW).
        """
        meta = DISEASE_METADATA.get(predicted_class, {
            "name": predicted_class,
            "category": "Unknown",
            "baseline_risk": RiskTier.REQUIRES_REVIEW,
            "clinical_significance": "Unknown class"
        })

        if quality_tier == QualityTier.POOR:
            return (
                RiskTier.REQUIRES_REVIEW,
                f"Image quality is POOR ({predicted_class} suggested with {confidence*100:.1f}% score). Re-capture under optimal lighting and focus before clinical review."
            )

        if confidence < 0.50:
            return (
                RiskTier.REQUIRES_REVIEW,
                f"Model confidence is low ({confidence*100:.1f}% for {meta['name']}). Manual clinical assessment by a dermatologist is strongly recommended."
            )

        if meta["baseline_risk"] == RiskTier.HIGH:
            return (
                RiskTier.HIGH,
                f"High-priority screening flag: Features consistent with {meta['name']} ({meta['category']}). Prompt face-to-face evaluation by a specialist is advised."
            )
        elif meta["baseline_risk"] == RiskTier.MODERATE:
            return (
                RiskTier.MODERATE,
                f"Moderate-priority screening flag: Features consistent with {meta['name']} ({meta['category']}). Clinical review recommended to prevent progression."
            )
        else:
            return (
                RiskTier.LOW,
                f"Low-priority screening flag: Features consistent with benign lesion ({meta['name']}). Continue standard observation for any changes in size, shape, or color."
            )
