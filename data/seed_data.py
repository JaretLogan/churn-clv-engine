"""
seed_data.py — Generates realistic customer + transaction data
and inserts it into PostgreSQL for model training.
"""

import os, random, math
from datetime import date, timedelta
import psycopg2
from psycopg2.extras import execute_batch

random.seed(42)

DB_URL = os.environ.get("DATABASE_URL", "postgresql://churn:churn@localhost:5432/churndb")

REGIONS   = ["East", "West", "Central", "South"]
SEGMENTS  = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Technology", "Furniture", "Office Supplies"]
PRODUCTS = {
    "Technology":      ["Laptop", "Monitor", "Keyboard", "Webcam", "Headset", "USB Hub"],
    "Furniture":       ["Chair", "Desk", "Bookshelf", "Filing Cabinet", "Lamp"],
    "Office Supplies": ["Pens", "Notebooks", "Stapler", "Printer Paper", "Binders"],
}
PRICE_RANGE = {
    "Laptop": (800, 1800), "Monitor": (200, 600), "Keyboard": (30, 150),
    "Webcam": (50, 200), "Headset": (40, 300), "USB Hub": (20, 80),
    "Chair": (150, 800), "Desk": (200, 1200), "Bookshelf": (80, 400),
    "Filing Cabinet": (100, 350), "Lamp": (25, 120),
    "Pens": (5, 30), "Notebooks": (8, 40), "Stapler": (10, 50),
    "Printer Paper": (20, 60), "Binders": (15, 45),
}

TODAY = date.today()
TRAIN_START = TODAY - timedelta(days=730)  # 2 years of history

def rand_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

def gen_customers(n=300):
    customers = []
    for i in range(n):
        cid = f"CUST-{i+1:04d}"
        customers.append((
            cid,
            f"Customer {i+1}",
            f"customer{i+1}@example.com",
            random.choice(REGIONS),
            random.choice(SEGMENTS),
            rand_date(TRAIN_START, TODAY - timedelta(days=90))
        ))
    return customers

def gen_transactions(customers):
    """
    Churned customers (30%): high early activity, stops 6+ months ago.
    Active customers (70%): regular purchases within last 90 days.
    """
    transactions = []
    churn_ids = set()

    for cid, *_ in customers:
        is_churned = random.random() < 0.30
        if is_churned:
            churn_ids.add(cid)

        if is_churned:
            # Churned: 1-5 orders, all older than 180 days ago
            n_orders = random.randint(1, 5)
            latest = TODAY - timedelta(days=180)
            earliest = TRAIN_START
        else:
            # Active: 3-20 orders, recent activity within 90 days
            n_orders = random.randint(3, 20)
            latest = TODAY
            earliest = TODAY - timedelta(days=365)

        order_dates = sorted([rand_date(earliest, latest) for _ in range(n_orders)])

        for odate in order_dates:
            cat = random.choice(CATEGORIES)
            product = random.choice(PRODUCTS[cat])
            lo, hi = PRICE_RANGE[product]
            amount = round(random.uniform(lo, hi), 2)
            qty = random.randint(1, 3)
            transactions.append((cid, odate, amount * qty, cat, product, qty))

    return transactions, churn_ids

def main():
    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor()

    print("Seeding customers...")
    customers = gen_customers(300)
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

if __name__ == "__main__":
    main()
