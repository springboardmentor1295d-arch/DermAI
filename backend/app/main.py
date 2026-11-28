from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.core.database import Base, engine, SessionLocal
from backend.app.core.security import get_password_hash
from backend.app.models import User, UserRole, ModelVersion
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.users import router as users_router
from backend.app.api.v1.cases import router as cases_router
from backend.app.api.v1.reports import router as reports_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.analytics import router as analytics_router
from backend.app.api.v1.notifications import router as notifications_router

setup_logging()


def sync_private_admin_account(db: Session):
    """
    Synchronizes the single private System Administrator account configured via
    environment variables (ADMIN_EMAIL / ADMIN_PASSWORD / ADMIN_FULL_NAME).
    Deactivates any stale/duplicate legacy administrator accounts so only one active admin exists.
    """
    admin_email = (settings.ADMIN_EMAIL or "").strip().lower()
    admin_password = (settings.ADMIN_PASSWORD or "").strip()
    admin_name = (settings.ADMIN_FULL_NAME or "System Administrator").strip()

    if not admin_email:
        return

    # Deactivate any other accounts holding ADMIN role
    extra_admins = (
        db.query(User)
        .filter(User.role == UserRole.ADMIN, User.email != admin_email)
        .all()
    )
    for extra in extra_admins:
        if extra.is_active:
            extra.is_active = False

    admin_user = db.query(User).filter(User.email == admin_email).first()
    if admin_user:
        admin_user.role = UserRole.ADMIN
        admin_user.full_name = admin_name
        admin_user.is_active = True
        if admin_password:
            admin_user.hashed_password = get_password_hash(admin_password)
    elif admin_password:
        admin_user = User(
            email=admin_email,
            hashed_password=get_password_hash(admin_password),
            full_name=admin_name,
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(admin_user)
    db.commit()


def sync_model_registry(db: Session):
    """
    Ensures ONLY the two verified production architectures (EfficientNet-B0 and MobileNetV3-Large)
    are registered with accurate weights paths and held-out evaluation metrics, while preserving
    whichever architecture is currently marked active.
    """
    from backend.app.services.ml_service import MODEL_SPECS, normalize_architecture_key
    from backend.app.api.v1.models import load_model_evaluation_payload

    existing = db.query(ModelVersion).all()
    by_arch = {}
    for row in existing:
        key = normalize_architecture_key(row.architecture)
        if key not in by_arch or row.is_active:
            by_arch[key] = row

    any_active = any(row.is_active for row in by_arch.values())

    for key in ("efficientnet_b0", "mobilenetv3_large"):
        spec = MODEL_SPECS[key]
        payload = load_model_evaluation_payload(key)
        tm = payload.get("test_metrics", {})

        model_row = by_arch.get(key)
        if model_row is None:
            model_row = ModelVersion(
                name=spec["full_name"],
                version_tag=spec["default_version_tag"],
                architecture=spec["display_name"],
                weights_path=str(spec["weights_path"]),
                test_accuracy=tm.get("overall_accuracy"),
                test_balanced_acc=tm.get("balanced_accuracy"),
                test_macro_f1=tm.get("macro_f1"),
                is_active=(not any_active and key == "efficientnet_b0"),
            )
            db.add(model_row)
            if model_row.is_active:
                any_active = True
        else:
            model_row.name = spec["full_name"]
            model_row.version_tag = spec["default_version_tag"]
            model_row.architecture = spec["display_name"]
            model_row.weights_path = str(spec["weights_path"])
            if tm:
                model_row.test_accuracy = tm.get("overall_accuracy")
                model_row.test_balanced_acc = tm.get("balanced_accuracy")
                model_row.test_macro_f1 = tm.get("macro_f1")

    db.commit()


def initialize_system_metadata(db: Session):
    """
    Initializes essential system metadata (single private admin + model registry).
    STRICT COMPLIANCE: Does NOT create any demo patients, clinicians, or sample analysis cases.
    """
    sync_private_admin_account(db)
    sync_model_registry(db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure schema and system metadata
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        initialize_system_metadata(db)
    finally:
        db.close()
    yield
    logger.info("Shutting down DermAI API server...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research and clinical decision-support platform for skin lesion classification "
        "and explainability from mobile images. Powered by PyTorch and EfficientNet-B0."
    ),
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Uploads directory for static image serving
app.mount("/artifacts/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Mount API Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(users_router, prefix=settings.API_V1_STR)
app.include_router(cases_router, prefix=settings.API_V1_STR)
app.include_router(reports_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(notifications_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health & Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }


@app.get("/ready", tags=["Health & Monitoring"])
def readiness_check():
    # Verify DB connectivity
    try:
        db = SessionLocal()
        db.query(User).first()
        db.close()
        db_ready = True
    except Exception:
        db_ready = False

    return {
        "status": "ready" if db_ready else "degraded",
        "database_connected": db_ready,
        "upload_directory_writable": settings.UPLOAD_DIR.exists()
    }
