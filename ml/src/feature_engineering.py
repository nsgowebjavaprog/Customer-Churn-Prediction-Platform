"""
feature_engineering.py
-----------------------
Defines the sklearn preprocessing pipeline (ColumnTransformer) used by
BOTH training and inference, so training/serving skew is impossible --
the exact same fitted transformer object is pickled and reused by the
FastAPI backend.

Techniques demonstrated (good interview talking points):
  - Numeric pipeline: median imputation + StandardScaler
  - Categorical pipeline: most-frequent imputation + OneHotEncoder
  - ColumnTransformer to combine heterogeneous feature types
  - Derived / engineered features (feature engineering, not just raw cols)
"""

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "avg_monthly_spend",     # engineered
    "tenure_years",          # engineered
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "contract",
    "internet_service",
    "online_security",
    "tech_support",
    "streaming_tv",
    "paperless_billing",
    "payment_method",
]

ALL_INPUT_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure_months",
    "contract", "internet_service", "online_security", "tech_support",
    "streaming_tv", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges", "num_support_calls",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features. Applied identically at train & inference time."""
    df = df.copy()

    # Guard against divide-by-zero for brand-new customers (tenure=0)
    safe_tenure = df["tenure_months"].replace(0, 1)
    df["avg_monthly_spend"] = (df["total_charges"].fillna(0) / safe_tenure).round(2)
    df["tenure_years"] = (df["tenure_months"] / 12).round(2)

    # senior_citizen sometimes arrives as int 0/1 -> keep as category-friendly str
    df["senior_citizen"] = df["senior_citizen"].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor
