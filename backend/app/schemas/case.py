from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.models.case import CaseStatus
from backend.app.models.image import QualityTier
from backend.app.schemas.prediction import PredictionResponse


class QualityCheckResponse(BaseModel):
    id: str
    blur_score: float
    brightness_mean: float
    contrast_std: float
    resolution_ok: int
    quality_tier: QualityTier
    advisory_flags: List[str]
    checked_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CaseImageResponse(BaseModel):
    id: str
    case_id: str
    image_url: str
    preprocessed_image_url: Optional[str] = None
    original_filename: str
    file_size_bytes: int
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    quality_check: Optional[QualityCheckResponse] = None
    prediction: Optional[PredictionResponse] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CaseCreate(BaseModel):
    patient_ref: Optional[str] = Field(None, max_length=64)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=32)
    anatomical_site: Optional[str] = Field(None, max_length=64)
    clinical_notes: Optional[str] = None


class CaseUpdate(BaseModel):
    clinical_notes: Optional[str] = None
    status: Optional[CaseStatus] = None
    patient_ref: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=32)
    anatomical_site: Optional[str] = None


class CaseResponse(BaseModel):
    id: str
    user_id: str
    patient_ref: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    anatomical_site: Optional[str] = None
    clinical_notes: Optional[str] = None
    status: CaseStatus
    upload_status: Optional[str] = "uploaded"
    created_at: datetime
    updated_at: datetime
    images: List[CaseImageResponse] = []

    model_config = ConfigDict(from_attributes=True)
