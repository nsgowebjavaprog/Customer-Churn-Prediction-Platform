"""
config.py
---------
Central place for paths and settings. Uses pydantic-settings so values
can be overridden by environment variables (12-factor app style) --
this is exactly what interviewers expect to see instead of hard-coded paths.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# =========================================================
# APPLICATION SETTINGS
# =========================================================

class Settings(BaseSettings):

    # Application
    app_name: str = "Customer Churn Prediction API"

    # ML model paths
    churn_model_path: str = str(
        BASE_DIR / "ml" / "models" / "churn_model.joblib"
    )

    metadata_file_path: str = str(
        BASE_DIR / "ml" / "models" / "model_metadata.json"
    )

    # Database
    database_url: str = (
        f"sqlite:///{BASE_DIR / 'backend' / 'app' / 'churn_history.db'}"
    )

    # Maximum number of rows allowed in uploaded CSV
    max_upload_rows: int = 20000

    # CORS
    cors_origins: list[str] = ["*"]

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# =========================================================
# SETTINGS INSTANCE
# =========================================================

settings = Settings()











'''
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root


class Settings(BaseSettings):
    app_name: str = "Customer Churn Prediction API"
    model_path: str = str(BASE_DIR / "ml" / "models" / "churn_model.joblib")
    metadata_path: str = str(BASE_DIR / "ml" / "models" / "model_metadata.json")
    database_url: str = f"sqlite:///{BASE_DIR / 'backend' / 'app' / 'churn_history.db'}"
    max_upload_rows: int = 20000
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
'''