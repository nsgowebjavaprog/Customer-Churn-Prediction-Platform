"""
ml_service.py
-------------
Wraps the trained sklearn Pipeline (preprocessor + classifier) for
inference. Loaded ONCE at app startup (see main.py lifespan) and
reused for every request -- loading a joblib model per-request would
be a classic performance mistake to avoid mentioning you *didn't* make
in an interview.

Feature engineering parity:
  The exact same `engineer_features()` function used in ml/src/train.py
  is imported here, guaranteeing zero train/serve skew.
"""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings

# Reuse the SAME feature engineering code that produced the training data.
# model_path looks like  <root>/ml/models/churn_model.joblib
# so its grandparent's sibling "src" is  <root>/ml/src
ML_ROOT = Path(settings.model_path).resolve().parent.parent  # <root>/ml
ML_SRC_PATH = ML_ROOT / "src"
sys.path.append(str(ML_SRC_PATH))
from feature_engineering import engineer_features, ALL_INPUT_COLUMNS  # noqa: E402


class ChurnModelService:
    def __init__(self):
        self.pipeline = None
        self.metadata = {}

    def load(self):
        self.pipeline = joblib.load(settings.model_path)
        with open(settings.metadata_path) as f:
            self.metadata = json.load(f)
        return self

    @property
    def model_name(self) -> str:
        return self.metadata.get("best_model", "unknown")

    def _risk_bucket(self, prob: float) -> str:
        if prob < 0.33:
            return "Low"
        if prob < 0.66:
            return "Medium"
        return "High"

    def predict_one(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        return self._predict_df(df)[0]

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        return self._predict_df(df)

    def _predict_df(self, df: pd.DataFrame) -> list[dict]:
        df_fe = engineer_features(df)
        X = df_fe[ALL_INPUT_COLUMNS + ["avg_monthly_spend", "tenure_years"]]
        probs = self.pipeline.predict_proba(X)[:, 1]
        preds = self.pipeline.predict(X)

        results = []
        for pred, prob in zip(preds, probs):
            results.append({
                "churn_prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(prob), 4),
                "risk_level": self._risk_bucket(float(prob)),
                "model_used": self.model_name,
            })
        return results


# Singleton instance imported by main.py / routers
churn_service = ChurnModelService()
