import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class CaseStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REQUIRES_REVIEW = "requires_review"
    FAILED = "failed"


class Case(Base):
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    patient_ref = Column(String(64), nullable=True)  # Synthetic or de-identified code e.g. "PT-9481"
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)
    anatomical_site = Column(String(64), nullable=True)  # e.g. "Arm", "Face", "Back", "Chest"
    image_source = Column(String(64), default="mobile_upload", nullable=False)
    clinical_notes = Column(Text, nullable=True)
    status = Column(Enum(CaseStatus), default=CaseStatus.PENDING, nullable=False)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    user = relationship("User", back_populates="cases")
    images = relationship("CaseImage", back_populates="case", cascade="all, delete-orphan")
