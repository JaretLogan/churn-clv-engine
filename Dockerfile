FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/main.py         ./main.py
COPY model/train.py      ./train.py
COPY data/seed_data.py   ./seed_data.py

# Model artifact directory (mounted as volume or written here)
RUN mkdir -p /app/model

# Entrypoint: start API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
