import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, Enum
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.core.timezone import UTCDateTime


class UserRole(str, enum.Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.PATIENT, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
