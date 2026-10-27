from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model once, when the API starts (the file is in the models folder of the project)
model = joblib.load(Path(__file__).parent.parent / "models" / "churn_model.joblib")

app = FastAPI(title="Churn risk API")


class Customer(BaseModel):
    # The 19 numbers that describe one customer (the same features as in the notebooks)
    recency_days: float
    n_invoices: float
    n_purchase_days: float
    n_invoices_90d: float
    n_invoices_180d: float
    invoices_trend: float
    monetary_total: float
    median_invoice_value: float
    monetary_90d: float
    monetary_180d: float
    tenure_days: float
    avg_gap_days: float | None = None      # may be empty for a customer with only one purchase day
    overdue_ratio: float | None = None     # may be empty for a customer with only one purchase day
    avg_items_per_invoice: float
    avg_lines_per_invoice: float
    n_products: float
    n_return_lines: float
    return_rate: float
    is_uk: int                             # 1 = UK, 0 = other country

    # An example that is filled in automatically on the /docs page
    model_config = {"json_schema_extra": {"examples": [{
        "recency_days": 38, "n_invoices": 6, "n_purchase_days": 6, "n_invoices_90d": 1, "n_invoices_180d": 3,
        "invoices_trend": -1, "monetary_total": 3402.39, "median_invoice_value": 598.22, "monetary_90d": 584.91,
        "monetary_180d": 1603.68, "tenure_days": 500, "avg_gap_days": 75.0, "overdue_ratio": 0.5,
        "avg_items_per_invoice": 200, "avg_lines_per_invoice": 20, "n_products": 90, "n_return_lines": 0,
        "return_rate": 0.0, "is_uk": 0,
    }]}}


@app.get("/")
def home():
    return {"message": "Churn risk API. Send a customer to /predict (open /docs to try it)."}


@app.post("/predict")
def predict(customer: Customer):
    # One row with the features in the same order as in training (empty values become NaN)
    data = pd.DataFrame([customer.model_dump()])[model.feature_names_in_].astype(float)

    # Risk score = the predicted probability of churn
    risk = model.predict_proba(data)[0, 1]
    return {"churn_risk": round(float(risk), 3)}