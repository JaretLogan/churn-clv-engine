# Customer Churn & CLV Prediction Engine

**Stack:** Python · PostgreSQL · Scikit-Learn · FastAPI · Docker Compose

Takes raw customer transaction history, engineers RFM features, predicts churn probability with a trained Random Forest, calculates 12-month Customer Lifetime Value, and serves everything through a live REST API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Docker Compose                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  PostgreSQL  │◄───│   Trainer    │    │   FastAPI    │  │
│  │   (churndb)  │    │  seed_data   │    │    :8000     │  │
│  │              │    │  + train.py  │    │              │  │
│  │  customers   │    │              │    │  /predict    │  │
│  │  transactions│    │  RF Model ──►│───►│  /analytics  │  │
│  │  predictions │◄───│  predictions │    │  /customers  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                    ▲         │
│                        model_artifact        model_artifact │
│                           (volume)             (volume)     │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `seed_data.py` generates 300 customers + realistic transaction history → PostgreSQL
2. `train.py` pulls via SQL, engineers 8 RFM features, trains Random Forest (200 trees), writes predictions back to DB
3. FastAPI loads the serialized model on startup and serves live inference at `/predict`

---

## Quickstart

**Prerequisites:** Docker + Docker Compose

```bash
git clone https://github.com/JaretLogan/churn-clv-engine
cd churn-clv-engine

docker compose up --build
```

Wait ~30 seconds for training to complete. API is live at **http://localhost:8000**, interactive docs at **http://localhost:8000/docs**.

> The `trainer` container exits after one run — that's expected. Only the `api` container stays up.

---

## API

### `POST /predict`

Score any customer from raw transaction history. No stored customer_id required.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-NEW-001",
    "segment": "Corporate",
    "region": "East",
    "transactions": [
      {"order_date": "2023-06-10", "amount": 450.00, "category": "Technology"},
      {"order_date": "2023-09-22", "amount": 120.50, "category": "Office Supplies"},
      {"order_date": "2023-11-05", "amount": 899.99, "category": "Technology"}
    ]
  }'
```

```json
{
  "customer_id":  "CUST-NEW-001",
  "churn_prob":   0.7231,
  "churn_label":  "CHURNED",
  "risk_tier":    "HIGH",
  "clv_score":    342.18,
  "rfm": {
    "recency":    212,
    "frequency":  3,
    "monetary":   1470.49
  },
  "explanation": "Churn probability 72.3%. Key drivers: last purchase was 212 days ago, low purchase frequency."
}
```

### Other endpoints

| Endpoint | Description |
|---|---|
| `GET /customers/{id}/prediction` | Stored prediction for a trained customer |
| `GET /analytics/summary` | Risk tier breakdown across all 300 customers |
| `GET /analytics/top-risk?limit=10` | Top N highest churn-risk customers |

---

## ML Details

### Features

| Feature | Description |
|---|---|
| `recency` | Days since last purchase |
| `frequency` | Total number of orders |
| `monetary` | Total lifetime spend |
| `avg_order_value` | Mean spend per transaction |
| `std_order_value` | Spend volatility |
| `category_diversity` | Distinct product categories purchased |
| `tenure_days` | Days from first to last purchase |
| `purchase_velocity` | Orders per 30 days |

Categorical features (`segment`, `region`) are one-hot encoded to match training schema.

### Model

Random Forest (200 trees, max_depth=8, `class_weight="balanced"` for ~30% churn minority). ROC-AUC evaluated on 20% holdout each training run. Model serialized to `/app/model/churn_model.pkl`, shared between containers via Docker volume.

### Churn definition

Customer is labeled churned if most recent purchase > 180 days ago. Configurable via `CHURN_DAYS` in `train.py`.

### CLV formula

```
CLV = (Avg Monthly Revenue × Gross Margin × 12) / (1 + Monthly Discount Rate)

Avg Monthly Revenue = avg_order_value × purchase_velocity
Gross Margin        = 0.25  (configurable)
Annual Discount     = 0.10
```

---

## Retrain

```bash
docker compose run --rm trainer python train.py
```

The API picks up the updated model artifact from the shared volume on its next request.

---

## Project Structure

```
churn-clv-engine/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   └── requirements.txt
├── data/
│   └── seed_data.py         # Data generation + PostgreSQL seeding
├── model/
│   └── train.py             # Feature engineering + RF training + CLV
├── sql/
│   └── init.sql             # Schema (auto-runs on DB start)
├── Dockerfile
├── docker-compose.yml
└── README.md
```
