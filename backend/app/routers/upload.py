import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.ml_service import churn_service

router = APIRouter(prefix="/upload", tags=["File Upload"])

# The exact columns the trained model expects. Kept here (single source
# of truth for validation) and mirrored by ALL_INPUT_COLUMNS in
# ml/src/feature_engineering.py which drives the actual prediction.
REQUIRED_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure_months",
    "contract", "internet_service", "online_security", "tech_support",
    "streaming_tv", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges", "num_support_calls",
]


def _validate_csv(df: pd.DataFrame) -> list[str]:
    errors = []
    if df.empty:
        errors.append("The uploaded CSV has no data rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {', '.join(missing_cols)}")

    if len(df) > settings.max_upload_rows:
        errors.append(
            f"Too many rows ({len(df)}). Max allowed is {settings.max_upload_rows}."
        )

    numeric_cols = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            if non_numeric.any():
                errors.append(f"Column '{col}' contains non-numeric values.")
    return errors


@router.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV of customers, get back the SAME CSV with three new
    columns: churn_prediction, churn_probability, risk_level.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    errors = _validate_csv(df)
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    predictions = churn_service.predict_batch(df)
    pred_df = pd.DataFrame(predictions)

    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )


@router.post("/validate-csv")
async def validate_csv_only(file: UploadFile = File(...)):
    """Dry-run validation without predicting — used by the frontend
    to show a friendly 'format OK / not OK' message before the user
    commits to running predictions."""
    if not file.filename.lower().endswith(".csv"):
        return {"valid": False, "errors": ["Only .csv files are accepted."]}

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        return {"valid": False, "errors": [f"Could not parse CSV: {exc}"]}

    errors = _validate_csv(df)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "row_count": len(df),
        "columns_found": list(df.columns),
    }
