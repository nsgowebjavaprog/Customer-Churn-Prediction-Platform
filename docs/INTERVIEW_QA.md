# ChurnGuard — Interview Q&A + "How to Tell This Story"

This document has two parts:
1. **How to present this project in an interview** — a narrative script.
2. **50+ Q&A** organized by topic (ML, FastAPI/Pydantic, Docker, System
   Design, Behavioral) with model answers you can adapt in your own words.

---

## Part 1 — How to tell it in an interview

### The 90-second version (elevator pitch)

> "I built an end-to-end customer churn prediction platform — not just a
> notebook, but the full pipeline a startup would actually ship. I
> generated a realistic customer dataset, did EDA, engineered features,
> and trained two algorithms — Logistic Regression and Random Forest —
> tracking every run in MLflow so I could compare them on precision,
> recall, F1, and ROC-AUC instead of just accuracy, since the data's
> imbalanced. My script automatically picks the better model — in my run,
> Logistic Regression actually won. I served that model behind a FastAPI
> backend with Pydantic-validated endpoints: a single-prediction API, a
> CSV upload endpoint that validates the file format and returns a scored
> CSV, and full CRUD over prediction history backed by SQLAlchemy. I built
> a React frontend for all three, containerized everything with Docker
> and docker-compose, and wrote pytest tests covering the happy path and
> the validation failures. I can walk through any layer in depth."

### The longer walkthrough (if they say "go ahead, show me")

1. **Start with the problem, not the code.** "Churn prediction is one of
   the highest-ROI ML use cases for a subscription or SaaS business —
   acquiring a new customer costs 5-25x more than retaining one, so even
   a modestly accurate model that flags at-risk customers a few weeks
   early has real business value."
2. **Show the EDA insight first.** Open `eda_summary.json` or mention the
   churn-by-contract breakdown. "Month-to-month customers churn at over
   2x the rate of two-year contract customers — that alone tells a
   business 'push customers toward annual contracts' before you even
   build a model."
3. **Show the two-model comparison and MLflow.** Open `mlflow ui` if you
   can, or just show `model_metadata.json`. Explain *why* you tracked
   multiple metrics, not just accuracy, and why the simpler model won.
4. **Open Swagger UI (`/docs`).** Walk through `/predict`, `/upload`, and
   `/history` live. This is genuinely the most impressive 60 seconds of
   the whole demo because it's interactive and requires zero setup for
   them to see it work.
5. **Show the CSV upload flow end-to-end** — upload
   `ml/data/customer_churn.csv` and download the scored version.
6. **Show the React frontend** doing the same thing through a UI instead
   of curl/Swagger.
7. **Mention Docker last**, as the "and it's all one command to run":
   `docker compose up --build`.

### If asked "what was hardest / what would you do differently?"

Have a genuine answer ready, e.g.: "Getting train/serve parity right —
making sure the exact same feature engineering runs at training time and
inference time — took real thought. I solved it by having the FastAPI
service import the same `engineer_features()` function from the training
module rather than re-implementing it, so there's structurally no way for
them to drift apart. If I extended this further, I'd add SHAP-based
explainability so the API could say *why* a customer is flagged high
risk, and I'd move the SQLite database to Postgres with Alembic
migrations for a real production deployment."

This shows self-awareness and forward thinking — exactly what "seniority"
signals look like even for a fresher/early-career candidate.

---

## Part 2 — Q&A by topic

### A. Machine Learning fundamentals

**Q1. Why did you generate synthetic data instead of using a real
dataset?**
So the project runs anywhere with zero external downloads or licensing
concerns, while keeping the exact schema and category vocabulary of the
well-known Telco Churn dataset. I also built in a real, non-trivial
relationship between features and the churn label (via a weighted score
passed through a sigmoid) so the ML actually has genuine signal to learn,
not random noise — I can *prove* that, because the trained model's own
top features (contract type, tenure, support calls) match the features I
used to generate the labels.

**Q2. Walk me through your EDA. What did you look for?**
Row/column counts, the target's class balance (34% churn — imbalanced,
which changes which metrics matter), missing values (I found ~2% missing
in `total_charges` and handled it with median imputation rather than
dropping rows, since dropping would lose otherwise-valid customer data),
distribution of numeric features, and — most importantly — churn rate
broken down by categorical slices like contract type, which surfaced the
single strongest predictor before I even trained a model.

**Q3. Why two algorithms? Why these two specifically?**
I wanted one linear, highly interpretable baseline (Logistic Regression)
and one non-linear ensemble that can capture feature interactions (Random
Forest). Comparing them tells you whether the extra complexity of a
non-linear model is actually earning its keep on this data — in my run it
wasn't; Logistic Regression won on ROC-AUC. That's a more honest signal
than assuming "more complex = better."

**Q4. Why did the simpler model win?**
My hypothesis: I engineered a ratio feature (`avg_monthly_spend`) and most
of the true relationships in this dataset are fairly monotonic (more
support calls → more risk; longer tenure → less risk), which favors a
linear model. I also deliberately regularized the Random Forest
(`max_depth=8`) to avoid overfitting on a moderately sized dataset, which
limited how much non-linear structure it could exploit. With a larger,
messier real-world dataset, I'd expect the gap to narrow or reverse.

**Q5. Why not just report accuracy?**
With a 66/34 class split, a model that always predicts "No churn" scores
66% accuracy while being completely useless for the business goal (it
never flags an at-risk customer). I used precision, recall, F1, and
ROC-AUC instead. In a churn context, I'd argue **recall** matters more
than precision in most business framings — missing an at-risk customer
(false negative) is usually more costly than flagging a customer who was
fine (false positive), since a false positive just costs a retention
email or a small discount offer.

**Q6. What does `class_weight="balanced"` actually do?**
It re-weights the loss function inversely proportional to class
frequency, so the model is penalized more for getting the minority class
(churners) wrong. It's a much cheaper alternative to actually
oversampling/undersampling the data (e.g. SMOTE), and it's supported
natively by both LogisticRegression and RandomForestClassifier in
scikit-learn.

**Q7. What is ROC-AUC, in plain terms?**
If you picked one random churner and one random non-churner, ROC-AUC is
the probability your model would assign a higher churn probability to the
actual churner. A value of 0.5 is random guessing; 1.0 is a perfect
ranking. It's threshold-independent, which is useful because the "right"
probability cutoff for flagging someone as high-risk is a business
decision, not a modeling one.

**Q8. Why `StandardScaler` for numeric features?**
Logistic Regression is fit by gradient-based optimization, and features
on wildly different scales (e.g. `tenure_months` 0-72 vs
`monthly_charges` 18-120) can slow or destabilize convergence and distort
the regularization penalty (which treats all coefficients' magnitudes
comparably — unscaled features get unfairly penalized or unpenalized).
Random Forest doesn't strictly need scaling (splits are scale-invariant),
but scaling doesn't hurt it either, and using one shared preprocessing
pipeline for both models keeps the code simpler and the comparison fair.

**Q9. Why one-hot encode rather than label/ordinal encode the categorical
features?**
Most of these categories (contract type, payment method, internet
service) have no inherent order — encoding them as 0/1/2 would falsely
imply `Two year > One year > Month-to-month` numerically to a linear
model, which is a spurious signal it would try to fit a coefficient to.
One-hot encoding avoids inventing an ordinal relationship that doesn't
exist.

**Q10. What does `handle_unknown="ignore"` do and why does it matter in
production?**
If a category never seen during training shows up at inference time (a
new payment method gets added to the product, for example), the encoder
would normally raise an error. `handle_unknown="ignore"` instead encodes
it as all-zeros across that feature's one-hot columns, so the pipeline
degrades gracefully instead of crashing the API in production.

**Q11. How did you handle missing data?**
Median imputation for numeric columns (robust to skew, unlike mean),
most-frequent imputation for categorical columns — both fit *inside* the
`Pipeline`'s `ColumnTransformer`, so the imputation strategy is learned
only from training data and applied consistently to any new data at
inference time, avoiding data leakage from the test/inference set into
the imputation statistics.

**Q12. What is data leakage, and did you avoid it anywhere specific
here?**
Data leakage is when information from outside the training set (often
from the test/validation set, or from the future) improperly influences
training, producing metrics that look better than real-world performance.
I avoided it by: (a) fitting the imputer/scaler/encoder only on the
training split, never on the full dataset before splitting, and (b) using
`stratify=y` in the split so the imbalance ratio, not any leaked signal,
is what's preserved between train and test.

**Q13. What's the difference between `fit`, `transform`, and
`fit_transform`?**
`fit` learns parameters from data (e.g. the mean/std for scaling, or
which categories exist for one-hot encoding). `transform` applies
already-learned parameters to (possibly different) data. `fit_transform`
does both in one call, and is used on the training set; on the test/live
data, only `transform` (via `pipeline.predict()`, which internally calls
transform, not fit) should be applied — refitting on test/live data would
be a leakage bug.

**Q14. Why wrap everything in a scikit-learn `Pipeline` instead of doing
preprocessing "by hand" before calling `.fit()`?**
Three reasons: (1) it guarantees the exact same transformation sequence
runs at train and predict time — no risk of forgetting a step at
inference; (2) it lets me persist ONE object (`joblib.dump(pipeline)`)
instead of juggling a scaler, an encoder, and a model as three separate
files that have to stay in sync; (3) it composes cleanly with
`GridSearchCV`/`cross_val_score` if I want to tune hyperparameters later,
since scikit-learn treats the whole pipeline as one estimator.

**Q15. What's MLflow actually doing for you here that print statements
wouldn't?**
Persistence and comparability across runs and time. Print statements
disappear when the terminal closes; MLflow stores every run's parameters,
metrics, and the model artifact itself in a queryable backend (SQLite in
this project), viewable in a UI, comparable across dozens of runs without
me maintaining my own spreadsheet. It's also what lets someone else on a
team reproduce or audit exactly what hyperparameters produced a specific
metric months later.

### B. FastAPI / Pydantic / API design

**Q16. Why FastAPI instead of Flask?**
Automatic request validation via Pydantic type hints, automatic OpenAPI
docs generation (`/docs`, `/redoc`) with zero extra code, native async
support, and a clean dependency-injection system (`Depends`) for things
like database sessions — all of which I'd otherwise have to hand-roll or
add via extra libraries in Flask.

**Q17. What is Pydantic actually doing when a request comes in?**
It parses the raw JSON body against the declared model's field types and
constraints. If a field is missing, has the wrong type, or violates a
constraint (like `Field(ge=0)`), Pydantic raises a validation error that
FastAPI automatically turns into an HTTP 422 response with a precise,
per-field error message — before any of my route logic executes.

**Q18. Why did you use enums for fields like `contract` and
`payment_method` instead of plain strings?**
Two reasons: it restricts input to exactly the categories the model was
trained on (preventing a typo like `"Month to month"` without the hyphen
from silently reaching the model and behaving unpredictably), and it
auto-renders as a dropdown in Swagger UI, which is a nicer developer
experience for anyone testing the API.

**Q19. Explain the difference between a path parameter, a query
parameter, and a request body in your API.**
A path parameter is part of the URL itself and identifies a specific
resource (`/history/{record_id}` — the ID is the resource identifier). A
query parameter is optional/filtering/configuration info appended after
`?` (`/predict/?save_to_history=true&customer_id=CUST-1` — controls
behavior, doesn't identify a resource). A request body carries the actual
payload of data being created or evaluated (the customer's features in
`POST /predict/`). I use all three across this API deliberately to
demonstrate each pattern where it's the right fit.

**Q20. Why separate `PredictionRecordCreate`, `PredictionRecordUpdate`,
and `PredictionRecordResponse` instead of one schema?**
Each represents a different contract. `Create` is what's needed to make a
new record (and shouldn't include server-generated fields like `id` or
`created_at`). `Update` needs every field optional, since PATCH requests
send only what's changing. `Response` is what the client receives back
and includes server-generated fields. Conflating them would let a client
try to set `id` or `created_at` directly, which should never be
client-controlled.

**Q21. How does your PATCH endpoint implement a "partial update"?**
`PredictionRecordUpdate` makes every field `Optional`. In `crud.py`, I
call `update.model_dump(exclude_unset=True)`, which returns only the
fields the client actually included in the request (not fields that
happen to be `None` because they weren't sent). I then loop over just
those and `setattr()` them on the ORM object — untouched fields are left
exactly as they were.

**Q22. What is dependency injection, concretely, in `Depends(get_db)`?**
`get_db()` is a generator function that creates a database session,
`yield`s it (handing control back to the route with that session as an
argument), and then — after the route function returns — resumes to
close the session, even if the route raised an exception, since it's
wrapped in a `try/finally`. FastAPI manages calling this generator and
injecting its yielded value into any route that declares
`db: Session = Depends(get_db)`, so I never manually open/close a session
inside a route body.

**Q23. Why does `/upload/validate-csv` always return `200`, but
`/upload/predict-csv` returns `400` on invalid files?**
They serve different purposes. `validate-csv` is a *dry-run check*
intended to be called repeatedly by the frontend as the user picks
different files — treating "this file is invalid" as a client error
(400) would be semantically odd for something whose entire job is to
report validity; it's a successful check that happens to report
"invalid." `predict-csv` is the actual action endpoint — if you can't
fulfill the action because the input is malformed, `400 Bad Request` is
the correct HTTP semantic.

**Q24. How would you version this API if you needed to make a breaking
change?**
I'd introduce a URL prefix like `/v2/predict` (or use a header-based
versioning scheme) rather than mutating `/predict` in place, so existing
clients (like the deployed React frontend, or any third-party
integration) don't break the moment a new model schema or field set
ships. FastAPI makes this easy since routers are just mounted with
`app.include_router(router, prefix="/v2")`.

**Q25. What HTTP status codes does your API use, and why those
specifically?**
`200` for successful GET/POST returning data, `204 No Content` for a
successful DELETE (nothing meaningful to return), `400` for a malformed
request the client can fix (bad CSV, bad file type), `404` for a
history record that doesn't exist, and `422` for Pydantic validation
failures on the request body/params (FastAPI's default for schema
violations).

### C. Databases / CRUD

**Q26. Why SQLite instead of Postgres for this project?**
Zero setup — anyone cloning the repo can run it immediately with no
external database server. Because I used SQLAlchemy as an ORM rather than
raw SQL, switching to Postgres in production is purely a
`DATABASE_URL` change; none of `crud.py`, `models_db.py`, or the routers
would need to change.

**Q27. What would you change about this schema for a production system
handling millions of predictions?**
Add an index explicitly on `created_at` (for time-range queries) in
addition to the existing indexes; consider partitioning the table by
month if it grows very large; add a background job to archive/delete very
old records rather than keeping every row forever; and switch from
`offset`/`limit` pagination (which gets slower on large offsets) to
cursor-based pagination (e.g. "give me records after ID X").

**Q28. How does your pagination work, and what's a downside of the
approach you used?**
`list_prediction_records` uses SQL `OFFSET`/`LIMIT` based on `page` and
`page_size`. It's simple and fine at this scale, but `OFFSET` on a very
large table gets progressively slower because the database still has to
scan and discard all the skipped rows. Cursor-based pagination
(`WHERE id < last_seen_id ORDER BY id DESC LIMIT N`) avoids that and
would be the production-scale improvement.

**Q29. Why is `input_payload` stored as JSON in the history table?**
So the full original request that produced a prediction is preserved for
audit/debugging (e.g. "why did we flag this customer as high risk six
months ago?") without needing a rigid, ever-changing set of columns for
every possible input field — SQLite/most databases support a native JSON
column type, and SQLAlchemy maps it straightforwardly.

### D. Docker / DevOps

**Q30. Why does the frontend Dockerfile have two `FROM` statements?**
It's a multi-stage build. Stage 1 (`node:20-alpine`) installs npm
dependencies and runs `npm run build` to produce static HTML/CSS/JS.
Stage 2 (`nginx:1.27-alpine`) copies *only* that static output and serves
it — none of Node, npm, or `node_modules` end up in the final image. This
makes the production image dramatically smaller and reduces attack
surface (no unnecessary build tooling shipped to production).

**Q31. Why does `backend/Dockerfile` copy `requirements.txt` and run
`pip install` before copying the rest of the app code?**
Docker caches each layer of a build. If application code changes but
`requirements.txt` doesn't, Docker reuses the cached "pip install" layer
instead of reinstalling every dependency from scratch, making rebuilds
during development dramatically faster.

**Q32. Why does the backend service in `docker-compose.yml` build from
the project root instead of the `backend/` folder?**
Because the backend's Dockerfile needs to `COPY` files from `ml/models`
and `ml/src` (the trained model and shared feature-engineering code) in
addition to `backend/app` — Docker's build context has to include
everything a `COPY` instruction might reference, and Docker doesn't allow
`COPY`ing from outside the build context.

**Q33. What does the named volume `backend_db` do, and what happens
without it?**
It persists the SQLite database file across container restarts/rebuilds
by mounting a Docker-managed storage location at `/app/app` inside the
container. Without it, every `docker compose down` (which removes
containers) would silently wipe all prediction history, since the
database file only exists inside the ephemeral container filesystem.

**Q34. `depends_on` — what does it guarantee and what doesn't it
guarantee?**
It guarantees Docker starts the `backend` container before the
`frontend` container. It does **not** guarantee the backend has finished
loading the model and is actually ready to accept requests — for that
you'd add a proper `healthcheck` to the backend service and use
`depends_on: {backend: {condition: service_healthy}}` on the frontend.

**Q35. How would you deploy this to production (say, on AWS or GCP)?**
Push both images to a container registry (ECR/GCR), run them on a managed
container service (ECS Fargate / Cloud Run / a Kubernetes cluster),
replace SQLite with a managed Postgres instance (RDS/Cloud SQL), put the
frontend behind a CDN (CloudFront/Cloud CDN) instead of a single nginx
container, and add a load balancer + auto-scaling policy in front of the
backend for traffic spikes.

### E. System design / product thinking

**Q36. How would you scale the `/predict` endpoint if traffic spiked
10x?**
Since the model is loaded in memory once at startup and inference is
CPU-bound and fast (a Logistic Regression is essentially a dot product),
horizontal scaling — running more identical backend container replicas
behind a load balancer — is the simplest fix, since there's no shared
mutable state in the prediction path itself (the DB write for history is
the only shared-state concern, and SQLite would become the bottleneck
there — that's exactly when I'd migrate to Postgres).

**Q37. How would you monitor this model in production?**
Track prediction volume and latency (operational health), the
distribution of predicted probabilities over time (a sudden shift could
indicate data drift), and — if ground truth churn labels become available
later (e.g. a customer actually cancels a month later) — periodically
recompute real precision/recall/F1 against that ground truth and alert if
they degrade meaningfully from the training-time metrics.

**Q38. What's the business cost of a false negative vs a false positive
here, and how would that change your threshold?**
A false negative (missed churner) means losing a customer with zero
retention effort spent — potentially the full customer lifetime value. A
false positive (flagging a healthy customer) usually just costs a
retention email, a small discount, or a wasted support-team touchpoint —
much cheaper. That asymmetry argues for a lower probability threshold
than the default 0.5 for flagging "high risk," accepting more false
positives to catch more true churners — a business/product decision I'd
make together with whoever owns the retention budget, not purely a
modeling decision.

**Q39. If the product team asked for real-time (sub-100ms) predictions
embedded in a live dashboard, would anything here need to change?**
Not fundamentally — the model is already loaded in memory and inference
is fast. I'd add response-time monitoring to confirm it, possibly move
the SQLite history write to a background task (`BackgroundTasks` in
FastAPI, or a message queue) so the write doesn't block the response, and
make sure the frontend calls the API asynchronously rather than blocking
the UI thread while waiting.

**Q40. How would you actually validate that this model creates business
value, not just "good offline metrics"?**
Run an A/B test: give a treatment group of at-risk-flagged customers a
retention intervention (discount, outreach call) and compare their actual
churn rate over the following quarter against a control group that
receives no intervention. Offline ROC-AUC tells you the model can *rank*
risk well; it doesn't by itself prove that acting on that ranking reduces
real churn — that requires a live experiment.

### F. Behavioral / "tell me about a challenge"

**Q41. What was the most technically interesting decision in this
project?**
Making sure feature engineering was shared, not duplicated, between
training and serving — importing `engineer_features()` directly into
`ml_service.py` rather than re-writing equivalent logic in the backend.
It's a small decision, but it structurally prevents the most common real
production ML bug (training/serving skew), and I think decisions like
that — the boring plumbing that prevents future bugs — are what
distinguish "I trained a model" from "I can ship an ML product."

**Q42. What would you do differently if you had another week?**
Add SHAP-based explainability to the `/predict` response, add a third
model (XGBoost) to the comparison, and add a proper CI pipeline
(GitHub Actions) running the pytest suite and building both Docker images
on every push, so the whole thing is verified automatically rather than
by me remembering to run tests before a demo.

**Q43. How did you validate your own work, beyond "it ran without
errors"?**
I wrote a pytest suite that checks: the model actually loads at startup,
a valid prediction returns a sane response shape, an invalid payload
(negative charge) is correctly rejected with a 422, the full CRUD flow
(create via prediction → list → read → update → delete) works end to
end, and a CSV missing required columns is correctly rejected with a 400.
I also manually ran the CSV upload against the full generated dataset and
confirmed the downloaded file had the right row count and new columns
appended in the right place.

**Q44. Why does this matter for a startup specifically, versus a big
company?**
Startups usually don't have a dedicated MLOps team, a platform team, and
a frontend team all pre-built — an early ML hire is often expected to
touch the model, the API, and sometimes even the UI that surfaces it.
This project is deliberately built to demonstrate comfort across that
whole stack, not just the modeling notebook, because that's closer to
what the first few ML hires at a startup actually do day to day.

**Q45. What's a limitation of this project you'd be upfront about if
asked?**
The dataset is synthetic, so the absolute metric values (ROC-AUC ~0.72)
don't represent real-world churn-prediction difficulty — real customer
data is messier and often has weaker per-feature signal. I'd frame this
project as demonstrating *engineering rigor and process* (EDA →
comparison → tracking → serving → testing → containerizing), which
transfers directly to a real dataset, rather than claiming the specific
numbers would hold on real company data.

---

## Quick-reference: one-line answers for rapid-fire rounds

- **What's overfitting?** A model that fits training data (including its
  noise) so closely that it fails to generalize to new data.
- **What's regularization?** A penalty added to a model's loss function
  that discourages overly large coefficients, reducing overfitting.
- **What's the bias-variance tradeoff?** Simple models tend to have high
  bias (underfit) and low variance; complex models tend to have low bias
  and high variance (overfit) — the goal is the sweet spot that
  minimizes total error on unseen data.
- **What's a confusion matrix?** A 2x2 table of true positives, false
  positives, true negatives, and false negatives for a binary classifier.
- **What's the difference between `predict()` and `predict_proba()`?**
  `predict()` returns the final class label; `predict_proba()` returns
  the underlying probability for each class, which is what let me
  compute ROC-AUC and define custom risk buckets (Low/Medium/High).
- **What's an idempotent HTTP method?** One where repeating the same
  request produces the same result with no additional side effects —
  GET, PUT, and DELETE are idempotent; POST is not (each call to
  `/predict` with `save_to_history=true` creates a *new* history row).
- **Why `async def` in FastAPI routes even though scikit-learn is
  synchronous?** FastAPI runs sync route functions in a thread pool
  automatically, so blocking calls like model inference don't block the
  whole event loop; declaring routes `async def` is only necessary when
  you're doing actual async I/O (like an async database driver) inside
  them, which this project doesn't currently use.
