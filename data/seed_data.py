"""
seed_data.py — Generates realistic noisy customer + transaction data.

Noise sources added to simulate real-world messiness:
  1. Win-back customers: churned but made one recent small purchase
  2. Lapsing active customers: active but haven't bought in a while
  3. High-value churned customers: churned despite large historical spend
  4. Erratic active customers: irregular purchase patterns with long gaps
  5. Amount noise: random outliers and bulk orders
  6. Category drift: customers who switched categories over time
"""

import os, random, math
from datetime import date, timedelta
import psycopg2
from psycopg2.extras import execute_batch

random.seed(7)  # different seed for more varied data

DB_URL = os.environ.get("DATABASE_URL", "postgresql://churn:churn@localhost:5432/churndb")

REGIONS    = ["East", "West", "Central", "South"]
SEGMENTS   = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Technology", "Furniture", "Office Supplies"]
PRODUCTS   = {
    "Technology":      ["Laptop", "Monitor", "Keyboard", "Webcam", "Headset", "USB Hub"],
    "Furniture":       ["Chair", "Desk", "Bookshelf", "Filing Cabinet", "Lamp"],
    "Office Supplies": ["Pens", "Notebooks", "Stapler", "Printer Paper", "Binders"],
}
PRICE_RANGE = {
    "Laptop": (800, 1800), "Monitor": (200, 600), "Keyboard": (30, 150),
    "Webcam": (50, 200),   "Headset": (40, 300),  "USB Hub": (20, 80),
    "Chair": (150, 800),   "Desk": (200, 1200),   "Bookshelf": (80, 400),
    "Filing Cabinet": (100, 350), "Lamp": (25, 120),
    "Pens": (5, 30), "Notebooks": (8, 40), "Stapler": (10, 50),
    "Printer Paper": (20, 60), "Binders": (15, 45),
}

TODAY       = date.today()
TRAIN_START = TODAY - timedelta(days=730)

def rand_date(start, end):
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))

def gen_customers(n=500):  # increased to 500 for more training signal
    customers = []
    for i in range(n):
        cid = f"CUST-{i+1:04d}"
        customers.append((
            cid,
            f"Customer {i+1}",
            f"customer{i+1}@example.com",
            random.choice(REGIONS),
            random.choice(SEGMENTS),
            rand_date(TRAIN_START, TODAY - timedelta(days=60))
        ))
    return customers

def gen_transactions(customers):
    transactions = []
    churn_ids = set()

    for cid, *_ in customers:
        is_churned = random.random() < 0.32  # ~32% churn rate
        if is_churned:
            churn_ids.add(cid)

        # ── Noise Type 1: Win-back customers ─────────────────────────────────
        # 10% of churned customers made one small recent purchase — model
        # should still flag them as at-risk due to low frequency + low spend
        winback = is_churned and random.random() < 0.10

        # ── Noise Type 2: Lapsing active customers ────────────────────────────
        # 15% of active customers haven't bought in 120-200 days — they're
        # still "active" in the business definition but look churned by recency
        lapsing = not is_churned and random.random() < 0.15

        # ── Noise Type 3: Erratic active customers ────────────────────────────
        # 10% of active customers have very irregular patterns — long gaps
        # between occasional large purchases (e.g. annual equipment buyers)
        erratic = not is_churned and random.random() < 0.10

        # ── Build order dates based on customer type ──────────────────────────
        if winback:
            # Mostly old orders + one recent small one
            n_orders = random.randint(1, 4)
            old_dates = [rand_date(TRAIN_START, TODAY - timedelta(days=200))
                         for _ in range(n_orders)]
            recent_date = rand_date(TODAY - timedelta(days=45), TODAY)
            order_dates = sorted(old_dates + [recent_date])

        elif lapsing:
            # Active but last purchase was 120-200 days ago
            n_orders = random.randint(2, 8)
            latest  = TODAY - timedelta(days=random.randint(120, 200))
            earliest = TRAIN_START
            order_dates = sorted([rand_date(earliest, latest) for _ in range(n_orders)])

        elif erratic:
            # Few orders spread over 2 years with big gaps
            n_orders = random.randint(2, 5)
            order_dates = sorted([rand_date(TRAIN_START, TODAY) for _ in range(n_orders)])

        elif is_churned:
            # Standard churned: all orders older than 180 days
            n_orders = random.randint(1, 6)
            latest  = TODAY - timedelta(days=180)
            earliest = TRAIN_START
            order_dates = sorted([rand_date(earliest, latest) for _ in range(n_orders)])

        else:
            # Standard active: regular recent purchases
            n_orders = random.randint(4, 22)
            latest  = TODAY
            earliest = TODAY - timedelta(days=365)
            order_dates = sorted([rand_date(earliest, latest) for _ in range(n_orders)])

        # ── Build transactions for each order date ────────────────────────────
        for odate in order_dates:
            cat     = random.choice(CATEGORIES)
            product = random.choice(PRODUCTS[cat])
            lo, hi  = PRICE_RANGE[product]
            amount  = round(random.uniform(lo, hi), 2)
            qty     = random.randint(1, 3)

            # ── Noise Type 4: High-value churned customers ────────────────────
            # 8% of churned customers had large orders — CLV signal matters here
            if is_churned and random.random() < 0.08:
                amount = round(random.uniform(800, 2500), 2)
                qty    = 1

            # ── Noise Type 5: Bulk order noise ────────────────────────────────
            # 5% of any order is a bulk purchase — inflates monetary signal
            if random.random() < 0.05:
                qty = random.randint(5, 15)

            # ── Noise Type 6: Amount jitter ───────────────────────────────────
            # Add gaussian noise to prices to simulate discounts/markups
            jitter = random.gauss(0, lo * 0.05)
            amount = max(round(amount + jitter, 2), 1.0)

            transactions.append((cid, odate, amount * qty, cat, product, qty))

    return transactions, churn_ids

def main():
    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor()

    print("Clearing old data...")
    cur.execute("DELETE FROM churn_predictions")
    cur.execute("DELETE FROM transactions")
    cur.execute("DELETE FROM customers")
    conn.commit()

    print("Seeding customers...")
    customers = gen_customers(500)
    execute_batch(cur,
        "INSERT INTO customers (customer_id, name, email, region, segment, acquired_date) "
        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        customers
    )

    print("Seeding transactions...")
    transactions, churn_ids = gen_transactions(customers)
    execute_batch(cur,
        "INSERT INTO transactions (customer_id, order_date, amount, category, product_name, quantity) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        transactions
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done — {len(customers)} customers, {len(transactions)} transactions, "
          f"{len(churn_ids)} churned ({len(churn_ids)/len(customers)*100:.0f}%)")
    print("Noise applied: win-back customers, lapsing actives, erratic buyers, "
          "high-value churned, bulk orders, price jitter")

if __name__ == "__main__":
    main()
