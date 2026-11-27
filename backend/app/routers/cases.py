import os

import cv2
from PIL import Image as PILImage
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import settings
from ..database import get_db
from ..deps import STAFF, audit, get_case_or_404, get_current_user
from ..services import preprocess, quality, storage

router = APIRouter(tags=["cases"])
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
PILImage.MAX_IMAGE_PIXELS = 300_000_000   # allow large phone / DSLR photos


def _limit(rgb, m=2048):
    """Store a downscaled copy (longest side <= m) so huge photos do not slow the app."""
    h, w = rgb.shape[:2]
    return rgb if max(h, w) <= m else cv2.resize(rgb, None, fx=m / max(h, w), fy=m / max(h, w), interpolation=cv2.INTER_AREA)


def img_out(i):
    return dict(id=i.id, case_id=i.case_id, url=f"/media/{i.path}", source=i.source, quality_label=i.quality_label,
                quality=i.quality, captured_at=i.captured_at, n_analyses=len(i.analyses))


def case_out(c, full=False):
    d = dict(id=c.id, patient_code=c.patient_code, age=c.age, gender=c.gender, body_location=c.body_location,
             created_at=c.created_at, n_images=len(c.images))
    if full:
        d.update(medical_history=c.medical_history, allergies=c.allergies, clinical_notes=c.clinical_notes,
                 images=[img_out(i) for i in c.images])
    return d


@router.post("/cases", status_code=201)
def create_case(body: schemas.CaseIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    c = models.Case(owner_id=user.id, **body.model_dump())
    db.add(c); db.commit(); audit(db, user, "create_case", str(c.id))
    return case_out(c, True)


@router.get("/cases")
def list_cases(db: Session = Depends(get_db), user=Depends(get_current_user)):
    q = db.query(models.Case)
    if user.role not in STAFF:
        q = q.filter(models.Case.owner_id == user.id)
    return [case_out(c) for c in q.order_by(models.Case.id.desc()).all()]


@router.get("/cases/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return case_out(get_case_or_404(db, case_id, user), True)


@router.patch("/cases/{case_id}/notes")
def add_notes(case_id: int, body: schemas.NotesIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    c = get_case_or_404(db, case_id, user)
    c.clinical_notes = body.clinical_notes; db.commit(); audit(db, user, "notes", str(case_id))
    return case_out(c, True)


async def _read(file: UploadFile):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Unsupported image format (use JPEG/PNG/WebP)")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    try:
        return preprocess.load_rgb(data)
    except Exception:
        raise HTTPException(422, "File is not a valid image")


@router.post("/quality/check")
async def quality_check(file: UploadFile = File(...), user=Depends(get_current_user)):
    """Step 'Quality Check' of the capture workflow, before the user confirms the upload."""
    return quality.assess(await _read(file))


@router.post("/cases/{case_id}/images", status_code=201)
async def upload_image(case_id: int, file: UploadFile = File(...), source: str = Form("upload"),
                       allow_poor: bool = Form(False), db: Session = Depends(get_db), user=Depends(get_current_user)):
    c = get_case_or_404(db, case_id, user)
    rgb = await _read(file)
    q = quality.assess(rgb)
    if q["label"] == "Poor" and not allow_poor:
        raise HTTPException(422, detail={"message": "Poor image quality - please recapture", "quality": q})
    rel = storage.save_rgb(_limit(rgb), f"images/case_{c.id}")
    img = models.Image(case_id=c.id, path=rel, source=source, width=q["width"], height=q["height"],
                       quality_label=q["label"], quality=q)
    db.add(img); db.commit(); audit(db, user, "upload_image", str(img.id))
    return img_out(img)
