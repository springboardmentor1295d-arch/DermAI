from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from backend.app.models.prediction import RiskTier


class TopNItem(BaseModel):
    class_code: str
    class_name: str
    probability: float
    percentage_str: str
    raw_score: Optional[float] = None
    calibrated_score: Optional[float] = None


class Top3PredictionItem(BaseModel):
    class_code: str
    class_name: str
    raw_score: float
    calibrated_score: Optional[float] = None
    raw_percentage_str: str
    calibrated_percentage_str: Optional[str] = None


class ExplainabilityInfo(BaseModel):
    gradcam_overlay_url: Optional[str] = None
    gradcam_heatmap_url: Optional[str] = None
    roi_mask_url: Optional[str] = None
    roi_bbox: Optional[List[int]] = None
    roi_area_ratio: Optional[float] = None


class StructuredPredictionResponse(BaseModel):
    """
    Standardized Phase 5 prediction response schema.
    Strictly integrates the deployed EfficientNet-B0 inference engine.
    """
    case_id: Optional[str] = None
    predicted_class: str
    class_name: str
    raw_model_score: float
    calibrated_confidence: Optional[float] = None
    top_3_predictions: List[Top3PredictionItem]
    model_version: str
    inference_timestamp: str
    inference_status: str
    risk_tier: Optional[RiskTier] = None
    risk_rationale: Optional[str] = None
    explainability: Optional[ExplainabilityInfo] = None
    preprocessed_image_url: Optional[str] = None
    input_dimensions: Optional[List[int]] = None
    preprocessing: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """
    Case-embedded prediction response maintaining full backward compatibility
    while delivering Phase 5 calibrated confidence and Top-3 predictions.
    """
    id: str
    image_id: str
    case_id: Optional[str] = None
    predicted_class: str
    class_name: str
    confidence_score: float
    confidence_percentage_str: str
    raw_model_score: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    top_3_predictions: Optional[List[Top3PredictionItem]] = None
    top_n_probabilities: List[TopNItem]
    risk_tier: RiskTier
    risk_rationale: Optional[str] = None
    gradcam_url: Optional[str] = None
    gradcam_heatmap_url: Optional[str] = None
    roi_mask_url: Optional[str] = None
    preprocessed_image_url: Optional[str] = None
    model_version: Optional[str] = None
    inference_timestamp: Optional[str] = None
    inference_status: Optional[str] = "SUCCESS"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
