"""
config.py
---------
Central place for paths and settings. Uses pydantic-settings so values
can be overridden by environment variables (12-factor app style) --
this is exactly what interviewers expect to see instead of hard-coded paths.
"""

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
