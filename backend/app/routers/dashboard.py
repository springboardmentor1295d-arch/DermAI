from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..deps import audit, get_current_user, require_roles
from ..services import risk

router = APIRouter(tags=["dashboard"])


@router.get("/notifications")
def notifications(db: Session = Depends(get_db), user=Depends(get_current_user)):
    rows = db.query(models.Notification).filter_by(user_id=user.id).order_by(models.Notification.id.desc()).limit(50).all()
    return [dict(id=n.id, level=n.level, title=n.title, message=n.message, read=n.read, created_at=n.created_at) for n in rows]


@router.post("/notifications/{nid}/read")
def mark_read(nid: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    n = db.get(models.Notification, nid)
    if n and n.user_id == user.id:
        n.read = True; db.commit()
    return {"ok": True}


@router.get("/stats/user")
def stats_user(db: Session = Depends(get_db), user=Depends(get_current_user)):
    rows = (db.query(models.Analysis).join(models.Image).join(models.Case)
            .filter(models.Case.owner_id == user.id).order_by(models.Analysis.id.desc()).all())
    return dict(total=len(rows), recent=[dict(id=a.id, cls=a.predicted_class, name=risk.CLASS_NAMES.get(a.predicted_class),
                confidence=a.confidence, risk=a.risk, created_at=a.created_at) for a in rows[:8]])


@router.get("/stats/clinician")
def stats_clinician(db: Session = Depends(get_db), user=Depends(require_roles("clinician", "admin", "researcher"))):
    an = db.query(models.Analysis).all()
    mv = db.query(models.ModelVersion).filter_by(is_active=True).first()
    return dict(total_cases=db.query(models.Case).count(), total_analyses=len(an),
                needing_review=sum(a.needs_review for a in an),
                disease_distribution=dict(Counter(a.predicted_class for a in an)),
                avg_confidence=(sum(a.confidence for a in an) / len(an)) if an else None,
                confidence_hist=[sum(1 for a in an if i / 10 <= a.confidence < (i + 1) / 10 + (a.confidence == 1)) for i in range(10)],
                active_model=dict(name=mv.name, version=mv.version, metrics=mv.metrics) if mv else None,
                recent=[dict(id=a.id, case_id=a.image.case_id, cls=a.predicted_class, confidence=a.confidence, risk=a.risk,
                             needs_review=a.needs_review) for a in sorted(an, key=lambda x: -x.id)[:10]])


@router.get("/stats/admin")
def stats_admin(db: Session = Depends(get_db), user=Depends(require_roles("admin", "ml_engineer"))):
    usage = (db.query(models.ModelVersion.name, models.ModelVersion.version, func.count(models.Analysis.id))
             .outerjoin(models.Analysis).group_by(models.ModelVersion.id).all())
    logs = db.query(models.AuditLog).order_by(models.AuditLog.id.desc()).limit(25).all()
    ds = db.query(models.Analysis.predicted_class, func.count()).group_by(models.Analysis.predicted_class).all()
    return dict(users=db.query(models.User).count(), analyses=db.query(models.Analysis).count(),
                cases=db.query(models.Case).count(), prediction_stats=dict(ds),
                model_usage=[dict(name=n, version=v, count=c) for n, v, c in usage],
                recent_activity=[dict(user_id=l.user_id, action=l.action, detail=l.detail, at=l.created_at) for l in logs],
                platform_health="ok")


@router.get("/admin/models")
def list_models(db: Session = Depends(get_db), user=Depends(require_roles("admin", "ml_engineer", "clinician", "researcher"))):
    return [dict(id=m.id, name=m.name, version=m.version, arch=m.arch, path=m.path, metrics=m.metrics,
                 is_active=m.is_active, created_at=m.created_at) for m in db.query(models.ModelVersion).order_by(models.ModelVersion.id)]


@router.post("/admin/models", status_code=201)
def register_model(body: schemas.ModelRegisterIn, db: Session = Depends(get_db), user=Depends(require_roles("admin", "ml_engineer"))):
    m = models.ModelVersion(**body.model_dump(), is_active=False, metrics={}); db.add(m); db.commit()
    audit(db, user, "register_model", f"{m.name} v{m.version}"); return {"id": m.id}


@router.post("/admin/models/{mid}/activate")
def activate(mid: int, db: Session = Depends(get_db), user=Depends(require_roles("admin", "ml_engineer"))):
    """Deploy / roll back: simply activate any earlier version."""
    m = db.get(models.ModelVersion, mid)
    if not m:
        raise HTTPException(404, "Model not found")

    import torch
    try:
        checkpoints = [
            torch.load(p.strip(), map_location="cpu", weights_only=True)
            for p in m.path.split(",") if p.strip()
        ]
        if not checkpoints:
            raise ValueError("No checkpoint")
    except Exception:
        raise HTTPException(422, "Invalid model checkpoint")

    if any(ck.get("use_meta") for ck in checkpoints):
        raise HTTPException(403, "Metadata fusion model is research-only")

    db.query(models.ModelVersion).update({"is_active": False})
    m.is_active = True
    db.commit()
    audit(db, user, "activate_model", str(mid)); return {"active": mid}
