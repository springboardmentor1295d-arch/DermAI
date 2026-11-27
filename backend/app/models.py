import datetime as dt

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base

now = lambda: dt.datetime.utcnow()
ROLES = ["patient", "clinician", "researcher", "ml_engineer", "admin"]


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), default="")
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="patient")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=now)


class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    patient_code = Column(String(64), default="")      # de-identified code, never real identifiers
    age = Column(Integer, nullable=True)
    gender = Column(String(16), default="")
    medical_history = Column(Text, default="")
    allergies = Column(Text, default="")
    clinical_notes = Column(Text, default="")
    body_location = Column(String(64), default="")
    created_at = Column(DateTime, default=now)
    images = relationship("Image", back_populates="case", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), index=True)
    path = Column(String(512))
    source = Column(String(32), default="upload")      # camera | gallery | upload
    width = Column(Integer); height = Column(Integer)
    quality_label = Column(String(16)); quality = Column(JSON)
    captured_at = Column(DateTime, default=now)
    case = relationship("Case", back_populates="images")
    analyses = relationship("Analysis", back_populates="image", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    name = Column(String(128)); version = Column(String(32)); arch = Column(String(64))
    path = Column(String(1024))                        # comma-separated paths => ensemble
    metrics = Column(JSON); is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=now)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    status = Column(String(16), default="completed")
    predicted_class = Column(String(8)); confidence = Column(Float)
    top3 = Column(JSON); probs = Column(JSON)
    risk = Column(String(32)); needs_review = Column(Boolean, default=False)
    lesion_area_ratio = Column(Float); seg_method = Column(String(32))
    explain_method = Column(String(32))
    preprocessed_path = Column(String(512)); mask_path = Column(String(512)); cam_path = Column(String(512))
    duration_ms = Column(Integer); created_at = Column(DateTime, default=now)
    image = relationship("Image", back_populates="analyses")
    model_version = relationship("ModelVersion")


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    level = Column(String(16)); title = Column(String(128)); message = Column(Text)
    read = Column(Boolean, default=False); created_at = Column(DateTime, default=now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True); action = Column(String(64)); detail = Column(Text, default="")
    created_at = Column(DateTime, default=now)
