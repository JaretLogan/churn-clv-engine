# Customer Churn & CLV Prediction Engine

**Tech Stack:** Python · PostgreSQL · Scikit-Learn · FastAPI · Docker · Docker Compose

A production-style machine learning system that ingests customer transaction history, engineers RFM (Recency, Frequency, Monetary) features, trains a Random Forest classifier to predict customer churn probability, calculates Customer Lifetime Value (CLV), and exposes the model as a live RESTful API endpoint.

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
1. `seed_data.py` generates 300 customers + realistic transaction history, inserts into PostgreSQL
2. `train.py` pulls data via SQL, engineers 8 RFM features, trains a Random Forest (200 trees), writes predictions back to DB
3. FastAPI loads the serialized model on startup and serves live inference via `/predict`

---

## Quickstart

**Prerequisites:** Docker + Docker Compose installed.

```bash
# 1. Clone the repo
git clone https://github.com/JaretLogan/churn-engine
cd churn-engine

# 2. Start everything (DB → seed → train → API)
docker compose up --build

# 3. Wait ~30 seconds for training to complete, then hit the API
curl http://localhost:8000/
```

The API is live at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

> **Note:** The `trainer` container runs once and exits — that's expected. The `api` container stays running.

---

## API Endpoints

### `POST /predict`
Score any customer from raw transaction history. No customer_id required — works on new/unknown customers.

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

**Response:**
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

### `GET /customers/{id}/prediction`
Fetch the stored prediction for a trained customer.
```bash
curl http://localhost:8000/customers/CUST-0001/prediction
```

### `GET /analytics/summary`
Risk tier breakdown across all 300 customers.
```bash
curl http://localhost:8000/analytics/summary
```

### `GET /analytics/top-risk?limit=10`
Top N highest churn-risk customers.
```bash
curl http://localhost:8000/analytics/top-risk?limit=10
```

---

## ML Pipeline Details

### Feature Engineering (RFM + Derived)

| Feature              | Description                              |
|----------------------|------------------------------------------|
| `recency`            | Days since last purchase                 |
| `frequency`          | Total number of orders                   |
| `monetary`           | Total lifetime spend                     |
| `avg_order_value`    | Mean spend per transaction               |
| `std_order_value`    | Spend consistency (volatility)           |
| `category_diversity` | # of distinct product categories bought |
| `tenure_days`        | Days from first to last purchase         |
| `purchase_velocity`  | Orders per 30 days (frequency / tenure)  |

Categorical features (`segment`, `region`) are one-hot encoded to match training column schema.

### Churn Label Definition
A customer is labeled **churned** if their most recent purchase is > 180 days ago. This threshold is configurable via `CHURN_DAYS` in `train.py`.

### Model
- **Algorithm:** Random Forest Classifier (200 trees, max_depth=8)
- **Class balancing:** `class_weight="balanced"` handles the ~30% churn minority class
- **Evaluation:** ROC-AUC score printed on each training run
- **Serialization:** Pickled to `/app/model/churn_model.pkl`, shared between containers via Docker volume

### CLV Formula
```
CLV = (Avg Monthly Revenue × Gross Margin × Projected Months)
          / (1 + Monthly Discount Rate)

where:
  Avg Monthly Revenue = avg_order_value × purchase_velocity
  Gross Margin        = 0.25 (configurable)
  Projected Months    = 12
  Annual Discount Rate = 0.10
```

---

## Project Structure

```
churn-engine/
├── app/
│   ├── main.py              # FastAPI application + endpoints
│   └── requirements.txt     # Python dependencies
├── data/
│   └── seed_data.py         # Data generation + PostgreSQL seeding
├── model/
│   └── train.py             # RFM engineering + RF training + CLV
├── sql/
│   └── init.sql             # PostgreSQL schema (auto-runs on DB start)
├── Dockerfile               # App + trainer container
├── docker-compose.yml       # Orchestrates DB + trainer + API
└── README.md
```

---

## Retrain on New Data

To retrain the model after adding new transactions:

```bash
docker compose run --rm trainer python train.py
```

The API container automatically picks up the updated model artifact from the shared volume on its next request.

---
