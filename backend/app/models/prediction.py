import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class RiskTier(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    REQUIRES_REVIEW = "requires_review"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("case_images.id"), nullable=False, unique=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=True)
    
    predicted_class = Column(String(32), nullable=False)  # BCC, ACK, NEV, SEK, SCC, MEL
    confidence_score = Column(Float, nullable=False)
    top_n_probabilities = Column(JSON, nullable=False)  # e.g. [{"class": "BCC", "name": "Basal Cell Carcinoma", "score": 0.85}, ...]
    
    risk_tier = Column(Enum(RiskTier), default=RiskTier.LOW, nullable=False)
    risk_rationale = Column(String(512), nullable=True)
    
    gradcam_overlay_path = Column(String(512), nullable=True)
    roi_mask_path = Column(String(512), nullable=True)
    
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    image = relationship("CaseImage", back_populates="prediction")
    model_version = relationship("ModelVersion", back_populates="predictions")
