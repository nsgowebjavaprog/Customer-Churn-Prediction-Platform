"""
routers/predict.py
-------------------
Single-customer prediction endpoint.

Demonstrates:
  - Request body validated by Pydantic (schemas.CustomerFeatures)
  - Query parameter (`save_to_history`) with a default value
  - Path parameter usage on the companion GET-by-id endpoint
  - Dependency injection of the DB session (get_db)
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app import schemas, crud
from app.database import get_db
from app.ml_service import churn_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=schemas.PredictionResponse)
def predict_churn(
    features: schemas.CustomerFeatures,
    save_to_history: bool = Query(
        default=True, description="Persist this prediction to history (CRUD demo)"
    ),
    customer_id: str | None = Query(default=None, description="Optional customer identifier"),
    db: Session = Depends(get_db),
):
    """Predict churn for ONE customer from a JSON body."""
    payload = features.model_dump()
    result = churn_service.predict_one(payload)

    if save_to_history:
        crud.create_prediction_record(
            db,
            schemas.PredictionRecordCreate(
                customer_id=customer_id,
                input_payload=payload,
                **result,
            ),
        )

    return schemas.PredictionResponse(**result)


@router.get("/model-info")
def model_info():
    """Returns metadata about the currently loaded model (both algos compared)."""
    return churn_service.metadata
