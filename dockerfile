# ============================
# 1. Base image Python 3.11
# ============================
FROM python:3.11.9-slim

WORKDIR /app

# ============================
# 2. Install system packages
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 3. Copy requirements
# ============================
COPY requirements.txt .

# ============================
# 4. Install Python dependencies
# ============================
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================
# 5. Copy source code
# ============================
COPY src ./src

# ============================
# 6. Copy data (important !)
# ============================
COPY data/train.csv ./data/train.csv

# Créer aussi le dossier processed
RUN mkdir -p data/processed models

# ============================
# 7. Pipeline entrypoint
# ============================
CMD python src/data_process.py && \
    python src/train.py && \
    python src/eval.py
