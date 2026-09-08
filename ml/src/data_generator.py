import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_ROWS = 3000

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_churn.csv"


def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    customer_id = [f"CUST-{10000+i}" for i in range(n_rows)]
    gender = RNG.choice(["Male", "Female"], n_rows)
    senior_citizen = RNG.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = RNG.choice(["Yes", "No"], n_rows)
    dependents = RNG.choice(["Yes", "No"], n_rows, p=[0.3, 0.7])

    tenure_months = RNG.integers(0, 73, n_rows)  # 0-72 months
    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.24, 0.21]
    )
    internet_service = RNG.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20]
    )
    online_security = RNG.choice(["Yes", "No", "No internet service"], n_rows)
    tech_support = RNG.choice(["Yes", "No", "No internet service"], n_rows)
    streaming_tv = RNG.choice(["Yes", "No", "No internet service"], n_rows)
    paperless_billing = RNG.choice(["Yes", "No"], n_rows, p=[0.6, 0.4])
    payment_method = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_rows
    )

    # Monthly charges correlated loosely with internet service / streaming
    base_charge = RNG.normal(65, 20, n_rows).clip(18, 120)
    monthly_charges = np.round(base_charge, 2)
    total_charges = np.round(monthly_charges * tenure_months + RNG.normal(0, 50, n_rows), 2)
    total_charges = np.clip(total_charges, 0, None)

    num_support_calls = RNG.poisson(1.5, n_rows)

    # ---- Build a latent "churn probability" so the data has real signal ----
    # (this is what makes EDA / feature importance meaningful later)
    churn_score = (
        -0.04 * tenure_months
        + 0.015 * monthly_charges
        + np.where(contract == "Month-to-month", 1.4, 0)
        + np.where(contract == "One year", 0.3, 0)
        + np.where(internet_service == "Fiber optic", 0.5, 0)
        + np.where(payment_method == "Electronic check", 0.5, 0)
        + 0.25 * num_support_calls
        + np.where(tech_support == "No", 0.4, 0)
        + np.where(senior_citizen == 1, 0.2, 0)
        - np.where(partner == "Yes", 0.2, 0)
        + RNG.normal(0, 1.0, n_rows)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_score + 2.2))
    churn = (RNG.uniform(0, 1, n_rows) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "churn": churn_label,
        }
    )

    # Inject a few missing values on purpose (real-world messiness)
    missing_idx = RNG.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[missing_idx, "total_charges"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
    print(df["churn"].value_counts(normalize=True))
