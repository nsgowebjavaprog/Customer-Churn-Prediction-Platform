"""
test_api.py
-----------
Basic smoke tests using FastAPI's TestClient (built on httpx).
Run:
    cd backend
    pytest -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

# Using TestClient as a context manager triggers FastAPI's lifespan
# (startup/shutdown) events -- required so the ML model actually loads.
client = TestClient(app)
client.__enter__()

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 5,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.5,
    "total_charges": 450.0,
    "num_support_calls": 3,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_single():
    resp = client.post("/predict/?save_to_history=false", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_invalid_payload():
    bad = SAMPLE_CUSTOMER.copy()
    bad["monthly_charges"] = -50  # violates ge=0 constraint
    resp = client.post("/predict/", json=bad)
    assert resp.status_code == 422  # Pydantic validation error


def test_history_crud_flow():
    # Create via prediction
    resp = client.post("/predict/?save_to_history=true&customer_id=CUST-TEST", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200

    # List
    resp = client.get("/history/?page=1&page_size=5")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1
    record_id = resp.json()["items"][0]["id"]

    # Read one
    resp = client.get(f"/history/{record_id}")
    assert resp.status_code == 200

    # Update
    resp = client.patch(f"/history/{record_id}", json={"notes": "reviewed by analyst"})
    assert resp.status_code == 200
    assert resp.json()["notes"] == "reviewed by analyst"

    # Delete
    resp = client.delete(f"/history/{record_id}")
    assert resp.status_code == 204


def test_csv_upload_missing_columns():
    csv_content = b"gender,tenure_months\nFemale,5\n"
    resp = client.post(
        "/upload/predict-csv",
        files={"file": ("bad.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 400
