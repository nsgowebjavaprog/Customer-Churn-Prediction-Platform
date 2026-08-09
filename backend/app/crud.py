"""
crud.py
-------
The "C-R-U-D" layer: every DB operation lives here, kept separate from
the route handlers (routers/*.py). This separation is a classic
FastAPI best practice interviewers look for -- routers stay thin
(HTTP concerns only), crud.py owns persistence logic.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func

from app import models_db, schemas


def create_prediction_record(db: Session, record: schemas.PredictionRecordCreate) -> models_db.PredictionRecord:
    db_record = models_db.PredictionRecord(
        customer_id=record.customer_id,
        churn_prediction=record.churn_prediction,
        churn_probability=record.churn_probability,
        risk_level=record.risk_level,
        model_used=record.model_used,
        input_payload=record.input_payload,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


def get_prediction_record(db: Session, record_id: int) -> models_db.PredictionRecord | None:
    return db.query(models_db.PredictionRecord).filter(
        models_db.PredictionRecord.id == record_id
    ).first()


def list_prediction_records(
    db: Session, page: int = 1, page_size: int = 20, risk_level: str | None = None
):
    query = db.query(models_db.PredictionRecord)
    if risk_level:
        query = query.filter(models_db.PredictionRecord.risk_level == risk_level)
    total = query.count()
    items = (
        query.order_by(models_db.PredictionRecord.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def update_prediction_record(
    db: Session, record_id: int, update: schemas.PredictionRecordUpdate
) -> models_db.PredictionRecord | None:
    db_record = get_prediction_record(db, record_id)
    if not db_record:
        return None
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(db_record, field, value)
    db.commit()
    db.refresh(db_record)
    return db_record


def delete_prediction_record(db: Session, record_id: int) -> bool:
    db_record = get_prediction_record(db, record_id)
    if not db_record:
        return False
    db.delete(db_record)
    db.commit()
    return True


def churn_rate_stats(db: Session) -> dict:
    total = db.query(func.count(models_db.PredictionRecord.id)).scalar() or 0
    churned = db.query(func.count(models_db.PredictionRecord.id)).filter(
        models_db.PredictionRecord.churn_prediction == "Yes"
    ).scalar() or 0
    return {
        "total_predictions": total,
        "predicted_churn_count": churned,
        "predicted_churn_rate": round(churned / total, 4) if total else 0.0,
    }
