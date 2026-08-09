"""
train.py
--------
End-to-end training script:
  1. Load raw CSV
  2. Quick EDA summary (printed + saved to ml/models/eda_summary.json)
  3. Feature engineering + preprocessing pipeline
  4. Train TWO algorithms: Logistic Regression (baseline, interpretable)
     and Random Forest (non-linear, usually stronger)
  5. Evaluate both on a held-out test set with multiple metrics
  6. Log params/metrics/artifacts to MLflow for EVERY run
  7. Select the better model (by ROC-AUC) and persist it + the
     preprocessor + metadata to ml/models/ for the FastAPI backend to load

Run:
    python ml/src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import (  # noqa: E402
    engineer_features, build_preprocessor, ALL_INPUT_COLUMNS,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODELS_DIR = ROOT / "models"
MLFLOW_TRACKING_DIR = ROOT / "mlruns"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def run_eda(df: pd.DataFrame) -> dict:
    """Lightweight EDA — the kind you'd show in a notebook / interview."""
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "churn_rate": round(float((df["churn"] == "Yes").mean()), 4),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_describe": df[
            ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]
        ].describe().to_dict(),
        "churn_by_contract": (
            df.groupby("contract")["churn"]
            .apply(lambda s: round((s == "Yes").mean(), 3))
            .to_dict()
        ),
    }
    print("\n===== EDA SUMMARY =====")
    print(f"Rows: {summary['n_rows']}  |  Churn rate: {summary['churn_rate']*100:.1f}%")
    print("Churn rate by contract type:", summary["churn_by_contract"])
    print("Missing values per column:", {k: v for k, v in summary["missing_values"].items() if v > 0})
    return summary


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    # SQLite backend store (production-realistic; the old plain-folder
    # "file:" store is now maintenance-mode only in MLflow 2.14+)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR / 'mlflow.db'}")
    mlflow.set_experiment("customer-churn-prediction")

    df = load_data()
    eda_summary = run_eda(df)
    with open(MODELS_DIR / "eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2, default=str)

    df = engineer_features(df)
    X = df[ALL_INPUT_COLUMNS + ["avg_monthly_spend", "tenure_years"]]
    y = (df["churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_prob)

            mlflow.log_param("model_type", name)
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")

            cm = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {**metrics, "confusion_matrix": cm}
            fitted_pipelines[name] = pipe

            print(f"\n--- {name} ---")
            print(metrics)
            print("Confusion matrix:", cm)
            _ = report  # already logged via mlflow metrics; kept for interview discussion

    # ----- Model selection: pick the one with higher ROC-AUC -----
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n>>> Selected best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    joblib.dump(best_pipe, MODELS_DIR / "churn_model.joblib")
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "feature_columns": ALL_INPUT_COLUMNS,
        }, f, indent=2)

    print(f"Saved best model -> {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
