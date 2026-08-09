FROM python:3.12-slim

WORKDIR /app

# System deps needed by pandas/scikit-learn wheels build occasionally
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: build context is the PROJECT ROOT (see docker-compose.yml),
# not this backend/ folder, because we also need files from ml/.
COPY backend/app ./app

# Copy the trained ML artifacts + feature engineering code the backend needs
COPY ml/models /app/ml/models
COPY ml/src /app/ml/src

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
