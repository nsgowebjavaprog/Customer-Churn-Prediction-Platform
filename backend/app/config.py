from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):

    app_name: str = "Customer Churn Prediction API"

    churn_model_path: str = str(
        BASE_DIR / "ml" / "models" / "churn_model.joblib"
    )

    metadata_file_path: str = str(
        BASE_DIR / "ml" / "models" / "model_metadata.json"
    )

    database_url: str = (
        f"sqlite:///{BASE_DIR / 'backend' / 'app' / 'churn_history.db'}"
    )

    max_upload_rows: int = 20000

    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()