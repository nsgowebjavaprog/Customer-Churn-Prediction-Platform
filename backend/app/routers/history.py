from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session

from app import schemas, crud
from app.database import get_db

router = APIRouter(prefix="/history", tags=["Prediction History (CRUD)"])


@router.get("/", response_model=schemas.PaginatedHistory)
def list_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    risk_level: str | None = Query(default=None, description="Filter by Low/Medium/High"),
    db: Session = Depends(get_db),
):
    items, total = crud.list_prediction_records(db, page, page_size, risk_level)
    return schemas.PaginatedHistory(total=total, page=page, page_size=page_size, items=items)


@router.get("/stats/summary")
def stats_summary(db: Session = Depends(get_db)):
    return crud.churn_rate_stats(db)


@router.get("/{record_id}", response_model=schemas.PredictionRecordResponse)
def get_history_item(record_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    record = crud.get_prediction_record(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return record


@router.patch("/{record_id}", response_model=schemas.PredictionRecordResponse)
def update_history_item(
    record_id: int,
    update: schemas.PredictionRecordUpdate,
    db: Session = Depends(get_db),
):
    record = crud.update_prediction_record(db, record_id, update)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return record


@router.delete("/{record_id}", status_code=204)
def delete_history_item(record_id: int, db: Session = Depends(get_db)):
    deleted = crud.delete_prediction_record(db, record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return None
