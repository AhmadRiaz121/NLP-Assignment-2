# ── Base image ──
FROM python:3.11-slim

# ── Install runtime dependencies ──
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──
WORKDIR /app

# ── Copy your Python dependencies ──
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy your application code ──
COPY api/ ./api/

# ── Prepare models directory ──
RUN mkdir -p /models
VOLUME ["/models"]

# ── Expose port ──
EXPOSE 8000

# ── Healthcheck ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# ── Run Uvicorn ──
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]