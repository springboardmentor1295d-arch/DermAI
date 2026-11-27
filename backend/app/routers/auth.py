from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..deps import audit, get_current_user, require_roles
from ..security import create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
SELF_SERVICE = {"patient", "clinician", "researcher"}


@router.post("/register", status_code=201)
def register(body: schemas.RegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter_by(email=body.email).first():
        raise HTTPException(409, "Email already registered")
    if body.role not in SELF_SERVICE:
        raise HTTPException(403, "This role must be assigned by an administrator")
    u = models.User(email=body.email, full_name=body.full_name, role=body.role, hashed_password=hash_password(body.password))
    db.add(u); db.commit(); audit(db, u, "register", body.role)
    return {"id": u.id, "email": u.email, "role": u.role}


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    u = db.query(models.User).filter_by(email=form.username).first()
    if not u or not verify_password(form.password, u.hashed_password):
        raise HTTPException(401, "Incorrect email or password")
    audit(db, u, "login")
    return {"access_token": create_token(u.email, u.role), "token_type": "bearer", "role": u.role, "name": u.full_name}


@router.get("/me")
def me(u=Depends(get_current_user)):
    return {"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role}


@router.post("/users/{user_id}/role")
def set_role(user_id: int, role: str, db: Session = Depends(get_db), admin=Depends(require_roles("admin"))):
    if role not in models.ROLES:
        raise HTTPException(400, "Unknown role")
    u = db.get(models.User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    u.role = role; db.commit(); audit(db, admin, "set_role", f"{user_id}->{role}")
    return {"id": u.id, "role": u.role}
