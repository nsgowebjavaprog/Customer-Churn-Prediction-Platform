import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings

MODEL_PATH = Path(
    settings.churn_model_path
).resolve()

ML_ROOT = MODEL_PATH.parent.parent

ML_SRC_PATH = ML_ROOT / "src"

if not ML_SRC_PATH.exists():
    raise FileNotFoundError(
        f"ML source directory not found: {ML_SRC_PATH}"
    )

if str(ML_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(ML_SRC_PATH))


from feature_engineering import (  # noqa: E402
    engineer_features,
    ALL_INPUT_COLUMNS,
)

class ChurnModelService:
    """
    Handles loading the trained churn model and
    generating predictions.
    """

    def __init__(self):
        self.pipeline = None
        self.metadata = {}

   
    def load(self):
        """
        Load the trained sklearn Pipeline and metadata.

        This should be called ONCE during FastAPI startup.
        """

        model_path = Path(
            settings.churn_model_path
        ).resolve()

        metadata_path = Path(
            settings.metadata_file_path
        ).resolve()

        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Churn model not found at:\n{model_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at:\n{metadata_path}"
            )

        self.pipeline = joblib.load(
            model_path
        )


        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        print(
            f"[startup] Loaded model: {self.model_name}"
        )

        return self
    @property
    def model_name(self) -> str:
        """
        Return the model name stored in metadata.
        """

        return self.metadata.get(
            "best_model",
            "unknown",
        )


    @staticmethod
    def _risk_bucket(prob: float) -> str:
        """
        Convert churn probability into a risk category.

        0.00 - 0.32 -> Low
        0.33 - 0.65 -> Medium
        0.66 - 1.00 -> High
        """

        if prob < 0.33:
            return "Low"

        if prob < 0.66:
            return "Medium"

        return "High"

    def predict_one(
        self,
        payload: dict,
    ) -> dict:
        """
        Generate prediction for a single customer.
        """

        if self.pipeline is None:
            raise RuntimeError(
                "ML model is not loaded. "
                "Call churn_service.load() during "
                "application startup."
            )

        df = pd.DataFrame(
            [payload]
        )

        return self._predict_df(df)[0]


    def predict_batch(
        self,
        df: pd.DataFrame,
    ) -> list[dict]:
        """
        Generate predictions for multiple customers.
        """

        if self.pipeline is None:
            raise RuntimeError(
                "ML model is not loaded. "
                "Call churn_service.load() during "
                "application startup."
            )

        return self._predict_df(df)



    def _predict_df(
        self,
        df: pd.DataFrame,
    ) -> list[dict]:
        """
        Perform feature engineering and prediction.
        """

        if df.empty:
            return []

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        df_fe = engineer_features(df)

        # ----------------------------------------------------
        # Features expected by the trained pipeline
        # ----------------------------------------------------

        required_columns = (
            ALL_INPUT_COLUMNS
            + [
                "avg_monthly_spend",
                "tenure_years",
            ]
        )

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in required_columns
            if column not in df_fe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required features after "
                f"feature engineering: {missing_columns}"
            )

        X = df_fe[
            required_columns
        ]

        probs = self.pipeline.predict_proba(
            X
        )[:, 1]

        preds = self.pipeline.predict(
            X
        )


        results = []

        for pred, prob in zip(
            preds,
            probs,
        ):
            probability = float(prob)

            results.append(
                {
                    "churn_prediction": (
                        "Yes"
                        if int(pred) == 1
                        else "No"
                    ),
                    "churn_probability": round(
                        probability,
                        4,
                    ),
                    "risk_level": self._risk_bucket(
                        probability
                    ),
                    "model_used": self.model_name,
                }
            )

        return results

churn_service = ChurnModelService()