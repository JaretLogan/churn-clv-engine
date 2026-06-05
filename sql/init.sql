-- ============================================================
-- Churn & CLV Analytics Engine — Database Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    name            VARCHAR(100),
    email           VARCHAR(100),
    region          VARCHAR(50),
    segment         VARCHAR(50),
    acquired_date   DATE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) REFERENCES customers(customer_id),
    order_date      DATE NOT NULL,
    amount          NUMERIC(10,2) NOT NULL,
    category        VARCHAR(50),
    product_name    VARCHAR(100),
    quantity        INT DEFAULT 1,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id   SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) REFERENCES customers(customer_id),
    churn_prob      NUMERIC(5,4),
    clv_score       NUMERIC(10,2),
    rfm_recency     INT,
    rfm_frequency   INT,
    rfm_monetary    NUMERIC(10,2),
    risk_tier       VARCHAR(20),
    predicted_at    TIMESTAMP DEFAULT NOW()
);

-- Indexes for analytical query performance
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(order_date);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON churn_predictions(customer_id);
