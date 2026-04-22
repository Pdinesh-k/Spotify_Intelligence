FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-train XGBoost model so first request is fast
RUN python3 -c "from ml.model import load_models; load_models()" 2>/dev/null || true

# Render injects $PORT at runtime
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
