from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite:///./dermai.db"
    secret_key: str
    access_token_minutes: int = 480
    storage_dir: str = "./storage"
    model_dir: str = "./backend/models"
    admin_email: str = "admin@dermai.local"
    admin_password: str
    max_upload_mb: int = 60
    # Project-defined research protocol (NOT a clinical threshold): below this -> flag for clinical review
    low_confidence_threshold: float = 0.50


settings = Settings()
DISCLAIMER = ("AI-assisted analysis for screening/research and clinical decision support only. "
              "This is NOT a standalone medical diagnosis. Consult a qualified dermatologist.")
