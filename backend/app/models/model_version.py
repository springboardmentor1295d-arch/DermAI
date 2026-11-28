import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(128), nullable=False)
    version_tag = Column(String(64), unique=True, index=True, nullable=False)  # e.g. "v1.0.0-mobilenetv3"
    architecture = Column(String(128), nullable=False)  # e.g. "MobileNetV3-Large"
    weights_path = Column(String(512), nullable=False)
    
    # Evaluation metrics
    test_accuracy = Column(Float, nullable=True)
    test_balanced_acc = Column(Float, nullable=True)
    test_macro_f1 = Column(Float, nullable=True)
    metrics_summary = Column(JSON, nullable=True)
    
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    predictions = relationship("Prediction", back_populates="model_version")
