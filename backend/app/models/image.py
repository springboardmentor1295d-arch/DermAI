import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class QualityTier(str, enum.Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class CaseImage(Base):
    __tablename__ = "case_images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id"), nullable=False, index=True)
    storage_path = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(64), nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    case = relationship("Case", back_populates="images")
    quality_check = relationship("QualityCheck", uselist=False, back_populates="image", cascade="all, delete-orphan")
    prediction = relationship("Prediction", uselist=False, back_populates="image", cascade="all, delete-orphan")


class QualityCheck(Base):
    __tablename__ = "quality_checks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("case_images.id"), nullable=False, unique=True)
    blur_score = Column(Float, nullable=False)  # Laplacian variance
    brightness_mean = Column(Float, nullable=False)
    contrast_std = Column(Float, nullable=False)
    resolution_ok = Column(Integer, default=1, nullable=False)
    quality_tier = Column(Enum(QualityTier), default=QualityTier.GOOD, nullable=False)
    advisory_flags = Column(JSON, default=list, nullable=False)  # List of warnings e.g. ["blur_detected", "low_contrast"]
    checked_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    image = relationship("CaseImage", back_populates="quality_check")
