from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from . import models
from .database import get_db
from .security import decode_token

oauth2 = OAuth2PasswordBearer(tokenUrl="auth/login")
STAFF = ("clinician", "admin")


def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> models.User:
    p = decode_token(token)
    if not p:
        raise HTTPException(401, "Invalid or expired token")
    u = db.query(models.User).filter(models.User.email == p["sub"]).first()
    if not u or not u.is_active:
        raise HTTPException(401, "User inactive or not found")
    return u


def require_roles(*roles):
    def dep(u: models.User = Depends(get_current_user)):
        if u.role not in roles:
            raise HTTPException(403, "Insufficient permissions")
        return u
    return dep


def audit(db, user, action, detail=""):
    db.add(models.AuditLog(user_id=user.id if user else None, action=action, detail=detail)); db.commit()


def notify(db, user_id, level, title, message):
    db.add(models.Notification(user_id=user_id, level=level, title=title, message=message)); db.commit()


def get_case_or_404(db, case_id, user):
    c = db.get(models.Case, case_id)
    if not c:
        raise HTTPException(404, "Case not found")
    if user.role not in STAFF and c.owner_id != user.id:
        raise HTTPException(403, "Not your case")
    return c
