.PHONY: install install-dev data eda train serve test lint typecheck docker-build drift clean

PY ?= python3

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

data: ## Download + verify the raw dataset
	$(PY) -m src.data.ingestion

eda: ## Generate EDA report + figures
	$(PY) -m src.eda.explore

train: ## Full training pipeline (baselines, tuning, eval, artifact)
	$(PY) -m src.models.train

serve: ## Run the inference API locally
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test: ## Unit + integration tests
	pytest

lint:
	ruff check src tests

typecheck:
	mypy

docker-build:
	docker build -t telco-churn-api:latest .

drift: ## PSI drift report over recent prediction logs
	$(PY) -m src.monitoring.drift

clean:
	rm -rf reports/figures reports/evaluation data/processed
