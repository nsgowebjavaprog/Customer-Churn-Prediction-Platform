"""
models_db.py
------------
SQLAlchemy ORM model(s). Named "models_db" (not "models.py") to avoid
any confusion with the ML model files in ml/models/.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from app.database import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, nullable=True, index=True)
    churn_prediction = Column(String, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    input_payload = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
