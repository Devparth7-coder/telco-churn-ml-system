#!/usr/bin/env bash
# End-to-end offline run: ingest -> EDA -> train (includes eval + artifact).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/3 Ingesting and verifying dataset"
python3 -m src.data.ingestion

echo "==> 2/3 Running exploratory analysis"
python3 -m src.eda.explore

echo "==> 3/3 Training, evaluating, and serializing the model"
python3 -m src.models.train

echo "==> Done. Artifacts: models/  Reports: reports/"
