from typing import Optional

from pydantic import BaseModel, EmailStr


class RegisterIn(BaseModel):
    email: EmailStr; password: str; full_name: str = ""
    role: str = "patient"   # only patient/clinician/researcher self-service; admin/ml_engineer need an admin


class CaseIn(BaseModel):
    patient_code: str = ""; age: Optional[int] = None; gender: str = ""
    medical_history: str = ""; allergies: str = ""; clinical_notes: str = ""; body_location: str = ""


class NotesIn(BaseModel):
    clinical_notes: str


class ModelRegisterIn(BaseModel):
    name: str; version: str; arch: str; path: str
