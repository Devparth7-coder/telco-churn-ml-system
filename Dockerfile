# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Production image for the churn prediction API.
# Build: docker build -t telco-churn-api:latest .
# Run:   docker run --rm -p 8000:8000 telco-churn-api:latest
# The trained artifact (models/) is copied at build time; rebuild the image to
# ship a new model version (see CI gates).
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user (no secrets baked into the image).
RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first: stable layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + config + trained artifact.
COPY src ./src
COPY configs ./configs
COPY models ./models

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
