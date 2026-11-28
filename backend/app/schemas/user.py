from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from backend.app.models.user import UserRole

ALLOWED_REGISTRATION_ROLES = {UserRole.PATIENT, UserRole.CLINICIAN}


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=255)
    role: UserRole = UserRole.PATIENT

    @field_validator("role")
    @classmethod
    def validate_registration_role(cls, v: UserRole) -> UserRole:
        if v not in ALLOWED_REGISTRATION_ROLES:
            raise ValueError("Public registration is restricted to Patient and Clinician roles.")
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=128)


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
