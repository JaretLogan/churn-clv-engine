"""
main.py — FastAPI Churn & CLV Prediction API

Endpoints:
  GET  /                          Health check
  GET  /customers/{id}/prediction Fetch stored prediction for a customer
  POST /predict                   Score a customer from raw transaction history
  GET  /analytics/summary         Aggregate risk tier breakdown
  GET  /analytics/top-risk        Top 10 highest churn probability customers
"""

import os, pickle, math
from datetime import date, timedelta, datetime
from typing import List, Optional

import psycopg2
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

DB_URL     = os.environ.get("DATABASE_URL", "postgresql://churn:churn@localhost:5432/churndb")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/churn_model.pkl")
TODAY      = date.today()
AVG_MARGIN = 0.25
DISCOUNT_R = 0.10
PERIODS    = 12


# ── Startup: load model ───────────────────────────────────────────────────────

model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        model_cache["model"]        = artifact["model"]
        model_cache["feature_cols"] = artifact["feature_cols"]
        print(f"✓ Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠ Model not found at {MODEL_PATH} — /predict endpoint unavailable until training runs.")
    yield

app = FastAPI(
    title="Customer Churn & CLV Prediction API",
    description=(
        "Accepts a customer's transaction history, engineers RFM features, "
        "and returns churn probability + Customer Lifetime Value via a "
        "trained Random Forest classifier."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg2.connect(DB_URL)
    try:
        yield conn
    finally:
        conn.close()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Transaction(BaseModel):
    order_date:   str   = Field(..., example="2024-03-15", description="YYYY-MM-DD")
    amount:       float = Field(..., gt=0, example=249.99)
    category:     Optional[str] = Field(None, example="Technology")
    product_name: Optional[str] = Field(None, example="Laptop")
    quantity:     int   = Field(1, ge=1)

class PredictRequest(BaseModel):
    customer_id:  str              = Field(..., example="CUST-0001")
    segment:      Optional[str]    = Field(None, example="Corporate")
    region:       Optional[str]    = Field(None, example="East")
    transactions: List[Transaction]= Field(..., min_length=1)

class PredictionResponse(BaseModel):
    customer_id:  str
    churn_prob:   float = Field(..., description="Probability of churn (0–1)")
    churn_label:  str   = Field(..., description="CHURNED or ACTIVE")
    risk_tier:    str   = Field(..., description="HIGH / MEDIUM / LOW")
    clv_score:    float = Field(..., description="Projected 12-month CLV in USD")
    rfm: dict
    explanation:  str

class StoredPrediction(BaseModel):
    customer_id:  str
    churn_prob:   float
    risk_tier:    str
    clv_score:    float
    rfm_recency:  int
    rfm_frequency:int
    rfm_monetary: float
    predicted_at: datetime


# ── Feature builder (mirrors train.py) ───────────────────────────────────────

def build_features_from_request(req: PredictRequest) -> pd.DataFrame:
    txns = pd.DataFrame([t.model_dump() for t in req.transactions])
    txns["order_date"] = pd.to_datetime(txns["order_date"])
    txns["amount"]     = txns["amount"].astype(float)

    recency   = (pd.Timestamp(TODAY) - txns["order_date"].max()).days
    frequency = len(txns)
    monetary  = txns["amount"].sum()
    avg_val   = txns["amount"].mean()
    std_val   = txns["amount"].std() if len(txns) > 1 else 0.0
    diversity = txns["category"].nunique() if "category" in txns else 1
    tenure    = max((txns["order_date"].max() - txns["order_date"].min()).days, 1)
    velocity  = frequency / tenure * 30

    row = {
        "recency":           recency,
        "frequency":         frequency,
        "monetary":          monetary,
        "avg_order_value":   avg_val,
        "std_order_value":   std_val,
        "category_diversity":diversity,
        "tenure_days":       tenure,
        "purchase_velocity": velocity,
    }

    # One-hot encode segment / region to match training columns
    all_segments = ["Consumer", "Corporate", "Home Office"]
    all_regions  = ["Central", "East", "South", "West"]
    for s in all_segments:
        row[f"segment_{s}"] = 1 if req.segment == s else 0
    for r in all_regions:
        row[f"region_{r}"] = 1 if req.region == r else 0

    return pd.DataFrame([row]), {
        "recency": recency, "frequency": frequency, "monetary": round(monetary, 2)
    }

def compute_clv(avg_val, velocity):
    monthly = avg_val * velocity
    return round(monthly * AVG_MARGIN * PERIODS / (1 + DISCOUNT_R / 12), 2)

def explain(churn_prob, rfm):
    reasons = []
    if rfm["recency"] > 120:
        reasons.append(f"last purchase was {rfm['recency']} days ago")
    if rfm["frequency"] < 3:
        reasons.append("low purchase frequency")
    if rfm["monetary"] < 100:
        reasons.append("low total spend")
    if not reasons:
        reasons.append("strong recent engagement and spend history")
    return f"Churn probability {churn_prob:.1%}. Key drivers: {', '.join(reasons)}."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Churn & CLV Prediction API",
        "status":  "healthy",
        "model_loaded": "model" in model_cache,
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Score a customer from raw transaction history.
    Returns churn probability, risk tier, and 12-month CLV projection.
    """
    if "model" not in model_cache:
        raise HTTPException(503, "Model not loaded. Run training pipeline first.")

    model        = model_cache["model"]
    feature_cols = model_cache["feature_cols"]

    df, rfm = build_features_from_request(req)

    # Align columns to training set (fill missing OHE cols with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)

    churn_prob = float(model.predict_proba(df)[0, 1])
    churn_label = "CHURNED" if churn_prob >= 0.5 else "ACTIVE"
    risk_tier   = "HIGH" if churn_prob >= 0.7 else ("MEDIUM" if churn_prob >= 0.4 else "LOW")

    avg_val  = float(df["avg_order_value"].iloc[0])
    velocity = float(df["purchase_velocity"].iloc[0])
    clv      = compute_clv(avg_val, velocity)

    return PredictionResponse(
        customer_id=req.customer_id,
        churn_prob=round(churn_prob, 4),
        churn_label=churn_label,
        risk_tier=risk_tier,
        clv_score=clv,
        rfm=rfm,
        explanation=explain(churn_prob, rfm)
    )

@app.get("/customers/{customer_id}/prediction", response_model=StoredPrediction, tags=["Prediction"])
def get_stored_prediction(customer_id: str, db=Depends(get_db)):
    """Fetch the latest stored prediction for a given customer_id."""
    cur = db.cursor()
    cur.execute(
        """SELECT customer_id, churn_prob, risk_tier, clv_score,
                  rfm_recency, rfm_frequency, rfm_monetary, predicted_at
           FROM churn_predictions
           WHERE customer_id = %s
           ORDER BY predicted_at DESC LIMIT 1""",
        (customer_id,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"No prediction found for customer {customer_id}")
    cols = ["customer_id","churn_prob","risk_tier","clv_score",
            "rfm_recency","rfm_frequency","rfm_monetary","predicted_at"]
    return StoredPrediction(**dict(zip(cols, row)))

@app.get("/analytics/summary", tags=["Analytics"])
def summary(db=Depends(get_db)):
    """Aggregate churn risk tier breakdown across all customers."""
    cur = db.cursor()
    cur.execute("""
        SELECT risk_tier,
               COUNT(*)                    AS customer_count,
               ROUND(AVG(churn_prob)*100,2) AS avg_churn_pct,
               ROUND(AVG(clv_score),2)      AS avg_clv
        FROM churn_predictions
        GROUP BY risk_tier
        ORDER BY avg_churn_pct DESC
    """)
    rows = cur.fetchall()
    return {"risk_breakdown": [
        {"risk_tier": r[0], "customer_count": r[1],
         "avg_churn_pct": r[2], "avg_clv_usd": r[3]}
        for r in rows
    ]}

@app.get("/analytics/top-risk", tags=["Analytics"])
def top_risk(limit: int = 10, db=Depends(get_db)):
    """Top N customers by churn probability."""
    cur = db.cursor()
    cur.execute("""
        SELECT p.customer_id, c.name, c.segment, c.region,
               p.churn_prob, p.clv_score, p.risk_tier,
               p.rfm_recency, p.rfm_frequency, p.rfm_monetary
        FROM churn_predictions p
        JOIN customers c ON p.customer_id = c.customer_id
        ORDER BY p.churn_prob DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cols = ["customer_id","name","segment","region","churn_prob","clv_score",
            "risk_tier","rfm_recency","rfm_frequency","rfm_monetary"]
    return {"top_risk_customers": [dict(zip(cols, r)) for r in rows]}
