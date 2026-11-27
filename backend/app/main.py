import glob
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import models
from .config import settings
from .database import Base, SessionLocal, engine
from .routers import analysis, auth, cases, dashboard
from .security import hash_password

log = logging.getLogger("dermai")
os.makedirs(settings.storage_dir, exist_ok=True)
Base.metadata.create_all(engine)

app = FastAPI(title="DermAI", version="1.0",
              description="Skin-lesion screening / clinical decision support. NOT a diagnostic device.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# DEV NOTE: /media is public static. For production serve images via signed URLs (S3/MinIO presigned) instead.
app.mount("/media", StaticFiles(directory=settings.storage_dir), name="media")
for r in (auth, cases, analysis, dashboard):
    app.include_router(r.router)


def sync_models(db):
    """Register every checkpoint found in model_dir; activate the first if none active."""
    known = {m.path for m in db.query(models.ModelVersion).all()}
    for p in sorted(glob.glob(os.path.join(settings.model_dir, "*.pt"))):
        if p in known:
            continue
        try:
            import torch
            ck = torch.load(p, map_location="cpu")
            m = dict(ck.get("metrics") or {})
            arch = ck["arch"]

            if ck.get("use_meta"):
                encoder = os.path.splitext(p)[0] + "_meta_encoder.joblib"

                if not os.path.isfile(encoder):
                    log.warning("Missing metadata encoder: %s", p)
                    continue

                m["research_only"] = True
                m["input_type"] = "Image + Clinical Metadata"
                m["evaluation"] = "PAD-UFES-20 test dataset"
        except Exception as e:
            log.warning("skip %s: %s", p, e); continue
        n = db.query(models.ModelVersion).count() + 1
        db.add(models.ModelVersion(name=os.path.basename(p)[:-3], version=f"1.{n - 1}", arch=arch, path=p, metrics=m))
    db.commit()
    if not db.query(models.ModelVersion).filter_by(is_active=True).first():
        first = next(
            (v for v in db.query(models.ModelVersion).order_by(
                models.ModelVersion.id
            ).all() if not (v.metrics or {}).get("research_only")),
            None
        )
        if first:
            first.is_active = True; db.commit()


@app.on_event("startup")
def startup():
    db = SessionLocal()
    try:
        if not db.query(models.User).filter_by(email=settings.admin_email).first():
            db.add(models.User(email=settings.admin_email, full_name="Administrator", role="admin",
                               hashed_password=hash_password(settings.admin_password)))
            db.commit()
        sync_models(db)
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
