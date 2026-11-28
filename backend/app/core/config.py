from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "DermAI: Skin Disease Detector"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "dermai_super_secret_development_key_change_in_production_32bytes_min"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # Private Single Administrator Account (configured exclusively via environment / .env)
    ADMIN_EMAIL: str = "sysadmin@dermai.org"
    ADMIN_PASSWORD: str = ""
    ADMIN_FULL_NAME: str = "System Administrator"
    
    # Database (Dual mode: default to SQLite for local dev, PostgreSQL for production)
    DATABASE_URL: str = "sqlite:///./dermai.db"
    
    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "artifacts" / "uploads"
    REPORT_DIR: Path = BASE_DIR / "artifacts" / "reports"
    ML_ARTIFACT_DIR: Path = BASE_DIR / "artifacts" / "ml"
    
    # Dataset path (preserves existing EDA files)
    PAD_DATASET_DIR: Path = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1")
    PAD_SPLIT_FILE: Path = Path(r"C:\DermAI_EDA\outputs\PAD-UFES-20\patient_level_split_stratified.csv")
    
    # Active ML model path (Strictly EfficientNet-B0 as sole deployed classifier)
    ACTIVE_MODEL_PATH: Path = ML_ARTIFACT_DIR / "efficientnet_b0_best.pth"
    CALIBRATION_PATH: Path = ML_ARTIFACT_DIR / "temperature_scaling.json"
    MODEL_METADATA_PATH: Path = ML_ARTIFACT_DIR / "model_metadata.json"
    
    # Rate limiting & upload limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".webp"]
    ALLOWED_IMAGE_MIME_TYPES: List[str] = ["image/png", "image/jpeg", "image/webp", "image/jpg"]
    
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure critical directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
settings.ML_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
