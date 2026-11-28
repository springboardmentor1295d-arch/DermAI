import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(128), nullable=False)  # e.g. "USER_LOGIN", "CASE_CREATED", "PREDICTION_EXECUTED"
    resource = Column(String(255), nullable=True)  # e.g. "case:d3e1...", "auth"
    details = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)
    timestamp = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
