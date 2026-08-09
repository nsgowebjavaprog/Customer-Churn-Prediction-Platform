# ChurnGuard — Complete Project Guide

**Customer Churn Prediction Platform: EDA → Feature Engineering → 2-Model
Comparison → MLflow Tracking → FastAPI (Pydantic + CRUD) → File Upload →
React Frontend → Docker**

This guide is written so you can (a) run the project end to end, (b)
understand every design decision, and (c) explain it confidently in an
interview. It is organized in the order you'd actually build the project.

---

## Table of Contents

1. Why this project stands out
2. System architecture
3. Part A — The ML pipeline (EDA, feature engineering, 2 algorithms, MLflow)
4. Part B — The FastAPI backend (Pydantic, CRUD, params)
5. Part C — File upload & batch prediction
6. Part D — The React frontend
7. Part E — Docker & docker-compose
8. Setup & run guide (local + Docker)
9. Testing
10. Extending the project further ("more things" to add)
11. Glossary of every technique used

---

## 1. Why this project stands out

Most "AI/ML portfolio projects" you'll see from other candidates are a
single Jupyter notebook that trains one model and prints an accuracy
score. That signals "I can call `.fit()`," not "I can ship a product."

This project is different because it demonstrates the **full lifecycle** a
startup actually cares about:

| Layer | What most candidates show | What this project shows |
|---|---|---|
| Data | One CSV, no cleaning shown | Synthetic data generator + documented EDA + injected missing values handled properly |
| Modeling | One algorithm, no comparison | **Two algorithms compared** on 5 metrics, best one **selected programmatically** |
| Experiment tracking | None | **MLflow** logging params/metrics/artifacts for every run |
| Serving | `model.predict()` in a notebook | A real **FastAPI** service with typed request/response schemas |
| Data persistence | None | **SQLAlchemy** + full **CRUD** (Create/Read/Update/Delete) history |
| Batch processing | None | **CSV upload with format validation** → batch predict → downloadable CSV |
| Frontend | None | A working **React** UI hitting the real API |
| Packaging | None | **Docker** + **docker-compose** for one-command startup |
| Tests | None | `pytest` suite covering happy-path and validation-failure cases |

When a Bengaluru AI/ML startup interviewer asks "walk me through a project
you built," this gives you a 10-minute narrative that touches almost every
skill in a typical job description: Python, pandas/NumPy, scikit-learn,
model evaluation, MLOps (MLflow), backend API design (FastAPI/Pydantic),
databases, frontend basics, and containerization — in one coherent story
instead of ten disconnected tutorials.

---

## 2. System architecture

```
                         ┌─────────────────────────┐
                         │        React SPA         │
                         │  (Single / Upload / Hist) │
                         └────────────┬─────────────┘
                                      │  fetch() JSON / multipart
                                      ▼
                         ┌─────────────────────────┐
                         │        FastAPI            │
                         │  ┌───────────────────┐   │
                         │  │ routers/predict.py │   │──┐
                         │  ├───────────────────┤   │  │
                         │  │ routers/upload.py  │   │  │  loads once at
                         │  ├───────────────────┤   │  │  startup
                         │  │ routers/history.py │   │  │
                         │  └───────────────────┘   │  ▼
                         │           │              │  ┌──────────────┐
                         │           ▼              │  │ ml_service.py│
                         │      crud.py (SQLAlchemy) │  │ (loads .joblib)
                         └────────────┬──────────────┘  └──────┬───────┘
                                      ▼                        │
                            ┌──────────────────┐               │
                            │  SQLite database  │               │
                            │ prediction_history │               │
                            └──────────────────┘               │
                                                                 ▼
                                                    ┌─────────────────────┐
                                                    │  ml/models/*.joblib  │
                                                    │  (trained Pipeline)  │
                                                    └──────────┬──────────┘
                                                                 ▲
                                                                 │ produced by
                                                    ┌─────────────────────┐
                                                    │     train.py         │
                                                    │  (2 algos + MLflow)  │
                                                    └──────────┬──────────┘
                                                                 ▲
                                                                 │
                                                    ┌─────────────────────┐
                                                    │ data_generator.py    │
                                                    │ + feature_engineering│
                                                    └─────────────────────┘
```

**Key architectural decision — shared feature engineering code.** The
`engineer_features()` function in `ml/src/feature_engineering.py` is
imported by BOTH `train.py` (offline, at training time) and
`ml_service.py` (online, at inference time). This eliminates the single
most common production ML bug: **training/serving skew**, where the
features computed at inference time subtly differ from the features the
model was trained on. This is a genuinely senior-level detail to raise in
an interview.

---

## 3. Part A — The ML Pipeline

### 3.1 `ml/src/data_generator.py` — synthetic data with real signal

Real interview datasets (Telco Churn, etc.) require downloads and have
licensing quirks. Instead, this script **generates** a dataset with the
exact same shape and column semantics, but with a mathematically
constructed "latent churn score" so the data has genuine, learnable signal
(short tenure + month-to-month contract + high support calls → higher
churn probability), plus realistic messiness:

- ~2% of `total_charges` values are set to `NaN` on purpose, so the
  pipeline has to handle **missing data** — a guaranteed interview topic.
- Categorical columns use the exact vocabulary of the real-world Telco
  churn dataset (`Month-to-month`, `Fiber optic`, `Electronic check`,
  etc.) so anyone with ML experience recognizes the domain instantly.

**How the label is generated (for your own understanding, not something
you need to memorize verbatim):** a weighted linear combination of tenure,
monthly charges, contract type, internet type, payment method, support
calls, and senior-citizen status produces a "churn score," which is passed
through a **sigmoid function** to get a probability, and a random draw
against that probability produces the final Yes/No label. This is
literally the *inverse* of what logistic regression does at training
time — which is a nice thing to point out if asked "how do you know your
model works?": the model's own coefficients (`contract`, `tenure`,
`support_calls`) end up recovering the same features used to generate the
labels.

### 3.2 `ml/src/feature_engineering.py` — the shared preprocessing contract

This module has three responsibilities:

1. **`engineer_features(df)`** — adds two derived features:
   - `avg_monthly_spend = total_charges / tenure_months` (guarded against
     divide-by-zero for brand-new customers with `tenure_months == 0`)
   - `tenure_years = tenure_months / 12`

   These aren't just "more columns" — they encode a *ratio* and a *unit
   conversion* that can be more linearly related to churn than the raw
   inputs, which specifically helps the Logistic Regression model (a
   linear model benefits far more from good feature engineering than a
   tree-based model does — a great interview point about *why* feature
   engineering still matters in the era of gradient boosting).

2. **`build_preprocessor()`** — returns a scikit-learn `ColumnTransformer`
   that:
   - Runs **numeric features** through `SimpleImputer(strategy="median")`
     then `StandardScaler()` (median imputation is robust to skew/outliers
     compared to mean imputation; scaling matters a lot for Logistic
     Regression since it uses gradient-based optimization).
   - Runs **categorical features** through
     `SimpleImputer(strategy="most_frequent")` then
     `OneHotEncoder(handle_unknown="ignore")`. The `handle_unknown="ignore"`
     flag is important: it means if a brand-new category shows up at
     inference time that the model never saw during training, the pipeline
     won't crash — it just encodes it as all-zeros. This is another
     "have you thought about production failure modes?" talking point.

3. **`ALL_INPUT_COLUMNS`** — the single source of truth for which raw
   columns the model expects. This same list is reused by the CSV upload
   validator, so the "what columns does my file need?" answer can never
   drift out of sync between training and the API.

### 3.3 `ml/src/train.py` — EDA, two algorithms, MLflow, model selection

Walking through what happens when you run `python ml/src/train.py`:

**Step 1 — Load & EDA (`run_eda`).** Computes and prints:
- Row/column counts and overall churn rate (imbalanced at ~34% — a good
  excuse to talk about **class imbalance** and why we use
  `class_weight="balanced"` in both models rather than plain accuracy).
- Missing value counts per column.
- `describe()` statistics on the numeric columns.
- **Churn rate broken down by contract type** — this is the single most
  informative slice in the dataset (month-to-month customers churn at
  roughly 2.4x the rate of two-year contract customers), and it's exactly
  the kind of one-line insight you'd screenshot for a slide in a real job.

The EDA summary is also saved as `ml/models/eda_summary.json` so it's
available for a report/dashboard even without opening Python again.

**Step 2 — Feature engineering + train/test split.** 80/20 split,
`stratify=y` so the churn rate is preserved identically in both the train
and test sets (crucial for imbalanced classification — without
stratification you could randomly get a test set with very few positive
examples, making metrics noisy and unreliable).

**Step 3 — Train TWO algorithms:**

- **Logistic Regression** (`max_iter=1000, class_weight="balanced"`) — a
  linear, highly interpretable baseline. Coefficients can be inspected to
  say *which features push churn probability up or down*, which matters a
  lot to a business stakeholder who wants "why did this customer churn?"
  not just "will they churn?"
- **Random Forest** (`n_estimators=200, max_depth=8,
  class_weight="balanced"`) — a non-linear ensemble that can capture
  interactions (e.g. "high support calls AND fiber internet" being worse
  than either alone) that a linear model cannot.

Both are wrapped in the *same* two-step `Pipeline`
(`preprocessor → classifier`), which means calling `.fit(X_train, y_train)`
fits the preprocessor and the classifier together, and calling
`.predict(X_new)` on raw, unprocessed data automatically applies the exact
same transformations that were fit on the training data — no separate
"remember to scale the test set the same way" bookkeeping bugs possible.

**Step 4 — Evaluate on 5 metrics** (`evaluate()`): accuracy, precision,
recall, F1, and ROC-AUC. This project deliberately does **not** rely on
accuracy alone, since with a ~34% churn rate a lazy "always predict No"
model would already score ~66% accuracy while being completely useless.
Precision/recall/F1 and ROC-AUC are what you'd actually be graded on in an
interview if asked "why not just report accuracy?"

**Step 5 — Log everything to MLflow** (`mlflow.start_run()` per model):
- `mlflow.log_param(...)` for every hyperparameter of the classifier
  (`n_estimators`, `max_depth`, `class_weight`, etc.)
- `mlflow.log_metrics(...)` for all 5 evaluation metrics
- `mlflow.sklearn.log_model(...)` to store the *entire fitted pipeline*
  as a versioned MLflow artifact

This means every training run is reproducible and comparable later via
`mlflow ui` — you can literally show an interviewer a table of every run
you ever did, with metrics, without having kept your own spreadsheet.

**Step 6 — Model selection.** `best_name = max(results, key=lambda n:
results[n]["roc_auc"])` — the script doesn't just train two models and
leave you to eyeball which is better; it programmatically selects the
model with the higher ROC-AUC and persists **only that one** to
`ml/models/churn_model.joblib`, alongside `model_metadata.json` recording
both models' full metric sets (so you can always explain "why did I pick
this one and not the other?").

> In one real run of this exact code, **Logistic Regression won**
> (ROC-AUC 0.719 vs Random Forest's 0.702) — a genuinely useful fact to
> discuss: the engineered ratio feature (`avg_monthly_spend`) and mostly
> monotonic relationships in this dataset favored a linear model, and a
> shallow, heavily-regularized Random Forest (`max_depth=8`) didn't have
> enough room to find non-linear structure that outweighed that. In an
> interview, "the more complex model didn't win, and here's my hypothesis
> why" is a MUCH stronger answer than reciting "Random Forest is always
> better."

### 3.4 Why MLflow specifically (not just print statements)

- **Experiment comparison** — `mlflow ui` gives you a sortable table of
  every run, every metric, every hyperparameter, without any custom code.
- **Artifact storage** — the fitted pipeline itself is versioned and
  retrievable by run ID, not just the final "winner" `.joblib` file.
- **Reproducibility** — anyone can look at a past run and know exactly
  which hyperparameters produced which metrics.
- **Industry relevance** — MLflow (or an equivalent like Weights & Biases)
  is genuinely used at almost every ML-mature startup; naming it
  unprompted signals you've seen how real ML teams work, not just Kaggle.

---

## 4. Part B — The FastAPI Backend

### 4.1 Why FastAPI (vs Flask/Django) — the interview-ready answer

FastAPI was chosen because it gives three things "for free" that would
otherwise be hand-rolled:

1. **Automatic request validation via Pydantic** — a malformed request
   (wrong type, out-of-range number, invalid enum value) is rejected with
   a `422 Unprocessable Entity` and a precise error message, *before* your
   business logic even runs.
2. **Automatic interactive docs** — `/docs` (Swagger UI) and `/redoc` are
   generated from your type hints and Pydantic models with zero extra
   code. This is the first thing to show an interviewer live.
3. **Native async support + dependency injection** — the `Depends(get_db)`
   pattern used throughout this project is FastAPI's dependency injection
   system: it hands each request a fresh SQLAlchemy session and guarantees
   it's closed afterward, without manual `try/finally` in every route.

### 4.2 `app/config.py` — settings via environment variables

Uses `pydantic-settings.BaseSettings` so every configurable value (model
path, database URL, CORS origins, max upload rows) has a sane default but
can be overridden by an environment variable of the same name. This is the
**12-factor app** principle: configuration lives in the environment, not
hard-coded in source, so the exact same Docker image can run in
dev/staging/prod with different settings just by changing env vars in
`docker-compose.yml`.

### 4.3 `app/schemas.py` — Pydantic models (the validation layer)

Several Pydantic techniques are deliberately demonstrated here:

- **Enums for categorical fields** (`GenderEnum`, `ContractEnum`,
  `PaymentMethodEnum`, etc.) — only legal values are accepted, and this
  also makes the field render as a **dropdown** in Swagger UI automatically.
- **`Field(ge=..., le=...)` constraints** — e.g.
  `monthly_charges: float = Field(ge=0, le=1000)` rejects negative or
  absurd values with zero `if` statements in the route.
- **`model_config` with `json_schema_extra`** — populates the "Try it out"
  example payload in Swagger UI automatically.
- **Separate schemas per purpose** — `CustomerFeatures` (input) vs
  `PredictionResponse` (output) vs `PredictionRecordCreate` /
  `PredictionRecordUpdate` / `PredictionRecordResponse` for the CRUD API.
  This Create/Update/Response split means a client can never accidentally
  set fields like `id` or `created_at` — those only exist on the
  output-only Response schema.
- **`PredictionRecordUpdate` makes every field `Optional`** — this enables
  **PATCH** semantics: the client sends only fields it wants to change,
  and `model_dump(exclude_unset=True)` in `crud.py` picks out just those.

### 4.4 `app/database.py` + `app/models_db.py` — SQLAlchemy ORM

- `database.py` sets up the `engine`, `SessionLocal` factory, declarative
  `Base`, and the `get_db()` generator used as `Depends(get_db)` in every
  route touching the database.
- `models_db.py` defines `PredictionRecord` — named `models_db.py` (not
  `models.py`) specifically to avoid colliding with the *ML* models in
  `ml/models/`.
- **SQLite** keeps the project runnable with zero external services, but
  because SQLAlchemy abstracts the database, moving to **Postgres** in
  production is a one-line change to `DATABASE_URL` — nothing in
  `crud.py`, `models_db.py`, or the routers changes.

### 4.5 `app/crud.py` — the CRUD layer, kept separate from routes

Every DB operation (`create_prediction_record`, `get_prediction_record`,
`list_prediction_records`, `update_prediction_record`,
`delete_prediction_record`, `churn_rate_stats`) lives here, not inside
route handlers. Routes stay thin (HTTP concerns only); CRUD logic is
independently testable. `list_prediction_records` demonstrates
**pagination** (`offset`/`limit`) and **filtering** (`risk_level`).

### 4.6 `app/routers/predict.py` — single prediction endpoint

- **Request body** validated via `CustomerFeatures`.
- **Query parameters** — `save_to_history: bool = Query(default=True)` and
  `customer_id: str | None = Query(default=None)` demonstrate optional
  query params with defaults and inline OpenAPI descriptions.
- **`GET /predict/model-info`** returns which algorithm won and both
  models' metrics from `model_metadata.json` — turning the API itself
  into proof you compared two models.

### 4.7 `app/routers/history.py` — full CRUD, with path params

- **`GET /history/{record_id}`** / **`DELETE /history/{record_id}`** use
  `Path(..., ge=1)` — invalid IDs rejected before hitting the database.
- **`PATCH /history/{record_id}`** — partial update (e.g. attach an
  analyst note).
- **`GET /history/stats/summary`** — aggregate churn-rate stats.

### 4.8 `app/ml_service.py` — the inference layer

`ChurnModelService` loads **once** at FastAPI startup (via `lifespan` in
`main.py`), not per-request. It exposes `predict_one()` (used by
`/predict`) and `predict_batch()` (used by CSV upload, vectorized across
the whole DataFrame in one call). Both call the **same**
`engineer_features()` used at training time — the train/serve parity
guarantee.

### 4.9 `app/main.py` — wiring it all together

- **`lifespan` async context manager** creates DB tables and loads the ML
  model before the app accepts traffic.
- **CORS middleware** — required because React (port 3000) and FastAPI
  (port 8000) run on different origins.
- **`/health`** — a liveness/readiness endpoint reporting
  `model_loaded: true/false`, exactly what a container healthcheck would
  poll.

---

## 5. Part C — File Upload & Batch Prediction

This directly implements "give the input CSV, check the format, and if
correct, return the CSV with predicted values."

**`POST /upload/validate-csv`** (dry run, always `200`): checks file
extension, non-empty data, all 15 required columns present (checked
against the same `ALL_INPUT_COLUMNS` used by training/inference — no
drift possible), row count under `max_upload_rows` (20,000 default), and
that numeric columns actually contain numbers. Returns
`{"valid": bool, "errors": [...], "row_count": N, "columns_found": [...]}`
so the frontend can show a friendly pass/fail message before committing.

**`POST /upload/predict-csv`** (the real thing): re-validates; on failure
returns `HTTP 400` with itemized `validation_errors`. On success:
1. `churn_service.predict_batch(df)` — vectorized batch inference
2. `pd.concat([original_df, predictions_df], axis=1)` — appends
   `churn_prediction`, `churn_probability`, `risk_level` onto the
   *original* data, preserving customer IDs and all original columns
3. Streams the result back as a real downloadable `.csv` via
   `StreamingResponse` with a `Content-Disposition: attachment` header

This mirrors real "batch scoring" tools startups build: an analyst
exports a CSV from a CRM, uploads it, and downloads a scored version to
act on (e.g. "call the 50 highest-risk customers this week").

---

## 6. Part D — The React Frontend

Three tabs, one API client module (`src/api.js`), plain CSS — deliberately
simple so 100% of it is legible in a live walkthrough.

- **`api.js`** — every backend call lives here (`predictSingle`,
  `validateCsv`, `predictCsv`, `fetchHistory`, `deleteHistoryItem`,
  `fetchStats`), base URL from `process.env.REACT_APP_API_BASE_URL`.
- **`PredictionForm.js`** — a fully controlled React form, submits via
  `fetch`, renders a color-coded risk level (green/amber/red) — turning a
  raw probability into an at-a-glance business signal.
- **`FileUpload.js`** — pick a file → auto-validate → show pass/fail →
  enable "Predict & Download" only once valid → triggers a real browser
  download via a temporary `<a>` + `URL.createObjectURL(blob)`.
- **`ResultTable.js`** — paginated history list, aggregate stats from
  `/history/stats/summary`, and delete support (the "D" in CRUD, exposed
  in the UI).
- **`Navbar.js`** — simple tab switcher (no routing library needed for
  three views — a good "what would you add for v2" answer: React Router).

---

## 7. Part E — Docker & docker-compose

- **`backend/Dockerfile`** — copies `requirements.txt` and installs deps
  *before* copying app code (Docker layer caching: code changes don't
  force a dependency reinstall). Copies `ml/models` and `ml/src` into the
  image so the container is self-contained.
- **`frontend/Dockerfile`** — a **multi-stage build**: stage 1
  (`node:20-alpine`) builds the React app; stage 2 (`nginx:1.27-alpine`)
  serves only the static output. The final image contains no Node.js or
  `node_modules` — smaller, more secure than shipping the build toolchain.
- **`frontend/nginx.conf`** — falls back to `index.html` for unknown
  paths, required for any single-page app.
- **`docker-compose.yml`** — backend builds from the **project root**
  (not just `backend/`) because its Dockerfile also `COPY`s from `ml/`.
  Env vars (`MODEL_PATH`, `METADATA_PATH`, `DATABASE_URL`, `CORS_ORIGINS`)
  override internal paths — config-via-environment end to end. A named
  volume (`backend_db`) persists SQLite history across restarts.
  `depends_on: [backend]` controls start order only (not full readiness —
  a good "what would you improve" answer: `condition: service_healthy`).

---

## 8. Setup & Run Guide

### 8.1 Prerequisites
Python 3.10+, Node.js 18+ & npm, Docker + Docker Compose (optional), Git.

### 8.2 Train the model (must run first — backend loads this file)
```bash
cd ml
pip install -r requirements.txt
python src/data_generator.py      # -> ml/data/customer_churn.csv
python src/train.py               # -> ml/models/churn_model.joblib + metadata
```
Expected tail of output:
```
>>> Selected best model: logistic_regression (ROC-AUC=0.7189)
Saved best model -> ml/models/churn_model.joblib
```

### 8.3 Run the backend
```bash
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000/docs**.

### 8.4 Run the frontend
```bash
cd ../frontend
npm install
npm start
```
Open **http://localhost:3000**.

### 8.5 Run everything with Docker instead
```bash
# from the project root, after 8.2 has produced ml/models/churn_model.joblib
docker compose up --build
```
Backend: http://localhost:8000/docs · Frontend: http://localhost:3000

### 8.6 Inspect MLflow experiment history
```bash
cd ml
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
Open **http://localhost:5000**.

### 8.7 Try the API with curl
```bash
curl -X POST "http://localhost:8000/predict/?save_to_history=false" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No",
       "tenure_months":5,"contract":"Month-to-month","internet_service":"Fiber optic",
       "online_security":"No","tech_support":"No","streaming_tv":"Yes",
       "paperless_billing":"Yes","payment_method":"Electronic check",
       "monthly_charges":89.5,"total_charges":450.0,"num_support_calls":3}'
```

### 8.8 Try the CSV upload with curl
```bash
curl -X POST "http://localhost:8000/upload/predict-csv" \
  -F "file=@ml/data/customer_churn.csv" -o predictions_output.csv
```

---

## 9. Testing
```bash
cd backend
pytest -v
```
Covers: health check, single prediction happy path, out-of-range value
(expects `422`), full history CRUD flow (create → list → read → update →
delete), and CSV upload with missing required columns (expects `400`).

---

## 10. Extending the project further ("more things" to add)

- **Model registry** — `mlflow.register_model()` to formally promote a
  model from "staging" to "production."
- **A third algorithm** — add **XGBoost**/**LightGBM**; the `candidates`
  dict in `train.py` is built so a new entry is a one-line addition.
- **Hyperparameter tuning** — `GridSearchCV`/`Optuna`, logging every trial
  to MLflow.
- **SHAP explainability** — return *why* a customer was flagged high-risk,
  not just the probability.
- **Auth** — API key or JWT dependency protecting CRUD/upload endpoints.
- **CI/CD** — GitHub Actions running `pytest` and building Docker images
  on every push.
- **PostgreSQL** — swap `DATABASE_URL`, add `alembic` migrations.
- **React Router + a chart** — real URLs per tab, plus a churn-by-contract
  chart straight from `eda_summary.json`.
- **Rate limiting** — `slowapi` on `/predict` and `/upload`.
- **Drift detection** — monitor incoming feature distributions vs training
  distribution over time.

---

## 11. Glossary of every technique used

- **EDA** — summarizing a dataset's shape, missingness, and relationships
  before modeling.
- **Feature engineering** — deriving new inputs (`avg_monthly_spend`,
  `tenure_years`) that make patterns easier for a model to learn.
- **`ColumnTransformer` / `Pipeline`** — bundles preprocessing + model into
  one fit/predict unit, guaranteeing identical transforms at train & serve.
- **`class_weight="balanced"`** — re-weights the loss function to
  compensate for an imbalanced target.
- **ROC-AUC** — measures how well a model ranks positives above negatives
  across all thresholds; robust to class imbalance.
- **Precision / Recall / F1** — precision = correctness of positive
  predictions; recall = coverage of actual positives; F1 = harmonic mean.
- **MLflow** — experiment tracking & model registry; logs params, metrics,
  artifacts per run.
- **Pydantic** — data validation library FastAPI uses for request/response
  schemas and OpenAPI generation.
- **CRUD** — Create, Read, Update, Delete.
- **SQLAlchemy ORM** — maps Python classes to DB tables.
- **Dependency Injection (`Depends`)** — a route declares what it needs
  (e.g. a DB session) and the framework supplies it.
- **Multipart file upload** — HTTP mechanism for uploading files, handled
  via FastAPI's `UploadFile`.
- **`StreamingResponse`** — returns a file-like response with headers so
  the browser downloads it.
- **CORS** — browser security mechanism controlling cross-origin requests.
- **Docker multi-stage build** — multiple `FROM` statements so build tools
  don't end up in the runtime image.
- **`docker-compose`** — defines/runs multiple containers together.
- **12-factor config** — configuration via environment variables, not
  hard-coded in source.
