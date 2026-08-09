"""
main.py
-------
FastAPI application entrypoint.

Run locally (from backend/ directory):
    uvicorn app.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.ml_service import churn_service
from app.routers import predict, upload, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    Base.metadata.create_all(bind=engine)   # create SQLite tables if missing
    churn_service.load()                    # load the trained model ONCE
    print(f"[startup] Loaded model: {churn_service.model_name}")
    yield
    # ---- Shutdown ----
    print("[shutdown] Bye!")


app = FastAPI(
    title=settings.app_name,
    description=(
        "End-to-end MLOps demo: Customer Churn Prediction. "
        "Includes single prediction, batch CSV prediction, and full "
        "CRUD prediction history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(history.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": settings.app_name}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": churn_service.pipeline is not None}
