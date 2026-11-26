FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: ~1 GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY config/ config/
COPY scripts/download_model.py scripts/
RUN python -m scripts.download_model

COPY src/ src/
COPY demo_samples/ demo_samples/
COPY assets/test_metrics.json assets/
COPY app.py .

# Don't run the server as root
RUN useradd -m -u 1000 user
USER user

EXPOSE 8000
# One worker keeps a single model in memory; threads handle concurrent requests
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8000", "app:app"]
