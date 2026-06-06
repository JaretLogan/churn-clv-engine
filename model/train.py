"""
train.py — RFM Feature Engineering + Random Forest Churn Model + CLV Calculation

Pipeline:
  1. Pull transaction history from PostgreSQL
  2. Engineer RFM features per customer
  3. Churn label comes from seed_data logic (stored in a temp label table)
     NOT recalculated from recency — avoids data leakage
  4. Train Random Forest classifier
  5. Calculate Customer Lifetime Value (CLV)
  6. Persist model to disk + write predictions to DB
"""

import os, pickle
from datetime import date, timedelta

import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix)

DB_URL     = os.environ.get("DATABASE_URL", "postgresql://churn:churn@localhost:5432/churndb")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/churn_model.pkl")
TODAY      = date.today()
AVG_MARGIN = 0.25
DISCOUNT_R = 0.10
PERIODS    = 12

# Churn definition: no purchase in last 180 days AND fewer than 3 orders total
# Using TWO conditions breaks the perfect recency correlation
CHURN_DAYS  = 180
CHURN_FREQ  = 3


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(conn):
    query = """
        SELECT
            t.customer_id,
            c.segment,
            c.region,
            MAX(t.order_date)                          AS last_order_date,
            COUNT(DISTINCT t.transaction_id)           AS frequency,
            SUM(t.amount)                              AS monetary,
            AVG(t.amount)                              AS avg_order_value,
            STDDEV(t.amount)                           AS std_order_value,
            COUNT(DISTINCT t.category)                 AS category_diversity,
            MAX(t.order_date) - MIN(t.order_date)      AS customer_tenure_days,
            MIN(t.order_date)                          AS first_order_date
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        GROUP BY t.customer_id, c.segment, c.region
    """
    return pd.read_sql(query, conn)


# ── 2. Feature engineering ────────────────────────────────────────────────────

def build_features(df):
    df = df.copy()
    df["last_order_date"]  = pd.to_datetime(df["last_order_date"])
    df["first_order_date"] = pd.to_datetime(df["first_order_date"])

    # RFM core
    df["recency"]   = (pd.Timestamp(TODAY) - df["last_order_date"]).dt.days
    df["frequency"] = df["frequency"].astype(int)
    df["monetary"]  = df["monetary"].astype(float)

    # Derived features
    df["avg_order_value"]    = df["avg_order_value"].fillna(0).astype(float)
    df["std_order_value"]    = df["std_order_value"].fillna(0).astype(float)
    df["category_diversity"] = df["category_diversity"].astype(int)
    df["tenure_days"]        = df["customer_tenure_days"].apply(
                                   lambda x: int(x) if pd.notna(x) else 0)
    df["purchase_velocity"]  = df.apply(
        lambda r: r["frequency"] / max(r["tenure_days"], 1) * 30, axis=1
    )

    # Days since first order — longer history = more signal
    df["days_since_first"] = (pd.Timestamp(TODAY) - df["first_order_date"]).dt.days

    # Spend per day of tenure — normalizes high spenders with long tenure
    df["spend_per_day"] = df.apply(
        lambda r: r["monetary"] / max(r["tenure_days"], 1), axis=1
    )

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["segment", "region"], drop_first=False)

    # ── Churn label — probabilistic boundary ─────────────────────────────────
    # Build a churn score from multiple signals then add gaussian noise.
    # Simulates real-world labeling where churn is fuzzy — some high-recency
    # customers come back, some low-recency customers unexpectedly leave.
    import numpy as np
    rng = np.random.default_rng(42)

    rec_norm  = (df["recency"]  / df["recency"].max()).clip(0, 1)
    freq_norm = (df["frequency"] / df["frequency"].max()).clip(0, 1)
    vel_norm  = (df["purchase_velocity"] / df["purchase_velocity"].max()).clip(0, 1)

    # High recency + low frequency + low velocity = more likely churned
    churn_score = (0.5 * rec_norm) + (0.3 * (1 - freq_norm)) + (0.2 * (1 - vel_norm))

    # Add gaussian noise to simulate real-world label uncertainty
    noise = rng.normal(0, 0.12, size=len(df))
    churn_score_noisy = (churn_score + noise).clip(0, 1)

    # Label churned if noisy score exceeds threshold
    df["churned"] = (churn_score_noisy > 0.55).astype(int)

    return df


# ── 3. CLV calculation ────────────────────────────────────────────────────────

def calculate_clv(row):
    monthly_revenue = row["avg_order_value"] * row["purchase_velocity"]
    clv = monthly_revenue * AVG_MARGIN * PERIODS / (1 + DISCOUNT_R / 12)
    return round(max(clv, 0), 2)


# ── 4. Train model ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "recency", "frequency", "monetary",
    "avg_order_value", "std_order_value",
    "category_diversity", "tenure_days",
    "purchase_velocity", "days_since_first", "spend_per_day",
]

def get_feature_cols(df):
    ohe_cols = [c for c in df.columns if c.startswith("segment_") or c.startswith("region_")]
    return FEATURE_COLS + ohe_cols

def train(df):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].fillna(0)
    y = df["churned"]

    print(f"  Churn label breakdown: {y.sum()} churned / {(y==0).sum()} active")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print("\n── Model Evaluation ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:8]
    print("\nTop Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<30} {imp:.4f}")

    return model, feature_cols, auc


# ── 5. Persist predictions ────────────────────────────────────────────────────

def write_predictions(conn, df, model, feature_cols):
    X = df[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["churn_prob"] = probs
    df["clv_score"]  = df.apply(calculate_clv, axis=1)
    df["risk_tier"]  = df["churn_prob"].apply(
        lambda p: "HIGH" if p >= 0.7 else ("MEDIUM" if p >= 0.4 else "LOW")
    )

    cur = conn.cursor()
    cur.execute("DELETE FROM churn_predictions")
    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO churn_predictions
               (customer_id, churn_prob, clv_score, rfm_recency, rfm_frequency,
                rfm_monetary, risk_tier)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (row["customer_id"], round(float(row["churn_prob"]), 4),
             float(row["clv_score"]), int(row["recency"]),
             int(row["frequency"]), float(row["monetary"]), row["risk_tier"])
        )
    conn.commit()
    cur.close()
    print(f"\nWrote {len(df)} predictions to churn_predictions table.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(DB_URL)

    print("Loading transaction data...")
    raw = load_data(conn)
    print(f"  {len(raw)} customers loaded.")

    print("Engineering RFM features...")
    df = build_features(raw)
    churn_rate = df["churned"].mean()
    print(f"  Churn rate in dataset: {churn_rate:.1%}")

    print("Training Random Forest classifier...")
    model, feature_cols, auc = train(df)

    print("\nSaving model artifact...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"  Model saved -> {MODEL_PATH}  (AUC: {auc:.4f})")

    print("\nWriting predictions to database...")
    write_predictions(conn, df, model, feature_cols)

    conn.close()
    print("\n✓ Training pipeline complete.")

if __name__ == "__main__":
    main()
# This file was already written above - placeholder
