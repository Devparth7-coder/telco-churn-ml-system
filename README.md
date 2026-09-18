# telco-churn-ml-system

Production-grade, end-to-end ML system that predicts **telecom customer churn** from the
IBM Telco Customer Churn dataset — from checksum-verified ingestion through tuned model
training, to a hardened FastAPI inference service with PSI drift monitoring.

Everything in this repository was **actually executed and verified** (training runs, test
suite, live API smoke tests, drift monitor); results below are measured, not projected.

---

## 1. Overview

| | |
|---|---|
| **Problem** | Predict which customers will churn (binary classification, one row = one customer snapshot) |
| **Data** | IBM Telco Customer Churn — 7,043 customers × 21 columns, SHA-256 pinned at ingestion |
| **Target** | `Churn` ∈ {Yes, No}; churn rate 26.5% (moderate imbalance) |
| **Primary metric** | F1 on the churn class (threshold-tuned); secondary: ROC-AUC, PR-AUC, Brier |
| **Deployment** | REST API (FastAPI/Uvicorn), single container, health/readiness probes |
| **Users** | Internal retention team + batch CRM scoring jobs |

**Measured final results** (held-out test set, 1,057 rows, first contact only at the end):

| Metric | Value |
|---|---|
| **F1 (churn)** | **0.6405** @ threshold 0.56 |
| Precision / Recall (churn) | 0.5714 / 0.7286 |
| ROC-AUC | 0.8521 |
| PR-AUC | 0.6617 |
| Brier | 0.1538 |
| Accuracy | 0.7833 |

Improvement over baselines: **+0.64 F1 vs majority class** (gate ≥ 0.05 passed);
+0.0095 F1 vs the tuned logistic-regression reference model. The honest reading: gradient
boosting and forests barely beat a well-regularized logistic model on this dataset; the
selected random forest won 5-fold CV and has simple operational characteristics.

## 2. Architecture

```
                        ┌────────────────────────────────────────────────────┐
                        │                    OFFLINE (training)               │
 Raw CSV (SHA-256       │  ingestion ─► pandera ─► stratified split          │
 pinned, checksummed) ──┤  train/val/test assignments persisted              │
                        │     │                                              │
                        │     ▼                                              │
                        │  baselines (majority, logreg)                      │
                        │     │                                              │
                        │     ▼                                              │
                        │  Optuna tuning per candidate (CV on TRAIN only)    │
                        │  lightgbm · hist_gb · random_forest · logreg       │
                        │     │                                              │
                        │     ▼                                              │
                        │  refit winner on train ─► threshold on VAL         │
                        │     │                                              │
                        │     ▼                                              │
                        │  TEST evaluated once ─► artifact + provenance      │
                        │  + leakage audit + drift reference stats           │
                        └─────────────────────────────┬──────────────────────┘
                                                      │ models/*.joblib + latest.json
                        ┌─────────────────────────────▼──────────────────────┐
                        │                    ONLINE (inference)               │
  HTTP ─► pydantic ─► rate limit ─► Predictor (pipeline+threshold) ─► JSON   │
              │                              │                               │
              │                              ▼                               │
              │                 prediction log (JSONL, hashed IDs)           │
              │                              │                               │
              │                              ▼                               │
              │                 PSI drift monitor (offline CLI / cron)       │
              └── 422 actionable errors; 429 backpressure; 401 auth          │
                        └─────────────────────────────────────────────────────┘
```

Key design rules enforced in code, not convention:

* **Training/inference parity**: the serialized artifact *is* the sklearn pipeline
  (cleaning → feature engineering → imputation/encoding → estimator). There is no second
  preprocessing implementation that could drift.
* **No leakage by construction**: preprocessing fits only on `train`; tuning sees only
  train CV folds; the threshold comes from `validation`; `test` is scored exactly once.
* **Fail-safe service**: strict request validation at the edge, ready-probe decoupled
  from liveness, model loaded once at startup.

## 3. Repository layout

```
telco-churn-system/
├── configs/config.yaml          # single non-secret config source
├── data/
│   ├── raw/                     # checksummed source CSV
│   ├── processed/               # split_assignments.csv (audit)
│   └── prediction_logs/         # JSONL served-prediction log (drift input)
├── models/                      # artifacts: *.joblib, *.meta.json, latest.json,
│                                # reference_stats.json (drift baseline)
├── reports/
│   ├── eda_report.md            # EDA conclusions
│   ├── figures/                 # EDA + evaluation figures
│   └── evaluation/              # metrics.json, training_report.md,
│                                # leakage_audit.md, drift_report.json
├── src/
│   ├── config.py                # YAML + env-override config (frozen dataclasses)
│   ├── logging_utils.py         # JSON structured logging
│   ├── data/
│   │   ├── ingestion.py         # download + SHA-256 verify + load
│   │   ├── schemas.py           # pandera: strict RAW contract, inference contract
│   │   ├── preprocessing.py     # CleaningTransformer + ColumnTransformer pipeline
│   │   └── splitting.py         # stratified split + persisted assignments
│   ├── features/engineering.py  # stateless, leakage-safe derived features
│   ├── models/
│   │   ├── baseline.py          # majority + logistic baselines
│   │   ├── tuning.py            # Optuna search spaces & objective
│   │   ├── train.py             # orchestrator CLI (python -m src.models.train)
│   │   ├── evaluate.py          # metrics, threshold sweep, figures
│   │   ├── registry.py          # artifact save/load + provenance metadata
│   │   └── predict.py           # Predictor inference wrapper
│   ├── api/
│   │   ├── schemas.py           # public API contract (pydantic)
│   │   └── main.py              # FastAPI app factory, middleware, endpoints
│   ├── monitoring/
│   │   ├── prediction_log.py    # JSONL logger (hashed customer ids)
│   │   └── drift.py             # PSI reference stats + drift CLI
│   └── eda/explore.py           # scripted EDA -> report + figures
├── tests/                       # 48 tests: data, features, model, API, robustness
├── scripts/run_full_pipeline.sh # ingest -> EDA -> train
├── .github/workflows/ci.yml     # lint -> test (incl. training gate) -> image build
├── Dockerfile                   # slim, non-root, healthcheck, pinned deps
├── docker-compose.yml
├── Makefile
├── requirements.txt / requirements-dev.txt   # fully pinned
├── pyproject.toml               # pytest / ruff / mypy config
└── .env.example
```

## 4. Dataset & features

**Source**: IBM Telco Customer Churn (public), downloaded by `src/data/ingestion.py` and
verified against the SHA-256 pinned in `configs/config.yaml`
(`16320c9c…55e91`). If upstream changes, ingestion fails loudly instead of silently
training on shifted data.

**Real-world quirk handled**: 11 customers with `tenure=0` carry a *whitespace-only*
`TotalCharges` cell (a single space character — not an empty string). The schema flags
only invalid values; these blanks are treated as missing and median-imputed. They are
**kept**, never dropped.

**Feature engineering** (stateless, inference-safe, all computed from snapshot fields):

| Feature | Definition | Why |
|---|---|---|
| `tenure_group` | binned tenure (0–12 / 13–24 / 25–48 / 49+ months) | churn risk is highly non-linear in tenure |
| `service_count` | phone + internet + 6 add-on services | bundle depth correlates with stickiness |
| `avg_monthly_charges` | `TotalCharges / (tenure + 1)` | price trend signal; +1 guards tenure=0 |
| `no_security_support` | internet customer w/o OnlineSecurity & TechSupport | known high-risk profile |

**Leakage audit** (full version: `reports/evaluation/leakage_audit.md`): `customerID`
dropped (identifier); `TotalCharges` kept (billing history, known at scoring time,
verified redundant-but-legitimate via EDA); no timestamps exist → stratified random split
valid; ids verified unique pre-split; preprocessing fitted on train only; a
single-feature AUC screen (flag ≥ 0.90) found no implausibly predictive column
(max: tenure 0.737).

## 5. ML methodology

1. **Baselines** — majority class (F1 = 0 by construction) and class-balanced logistic
   regression through identical preprocessing/CV (CV F1 0.6329).
2. **Candidates** — LightGBM, HistGradientBoosting, RandomForest, LogisticRegression,
   each tuned with seeded TPE Optuna (12–30 trials, median pruning), objective = mean
   out-of-fold churn-class F1 under 5-fold stratified CV on train only.
3. **Selection** — best CV F1, near-tie preference for simpler models (ε = 0.002):

   | Model | CV F1 | std |
   |---|---|---|
   | **random_forest** (selected) | **0.6388** | 0.0016 |
   | lightgbm | 0.6342 | 0.0014 |
   | logreg (tuned) | 0.6327 | 0.0010 |
   | hist_gb | 0.6320 | 0.0014 |

4. **Threshold** — 0.56 chosen on the validation split by F1 sweep (0.10–0.90),
   independent of hyperparameter tuning.
5. **Test** — scored once: F1 0.6405 / ROC-AUC 0.8521 (table above).

**Interpretability** — the artifact's forest exposes `feature_importances_`;
permutation importance + partial dependence are the recommended follow-ups and the
pipeline structure supports them without changes. Global importances from the forest:
`Contract`, `tenure`, `TotalCharges`, `MonthlyCharges` and `InternetService` dominate —
consistent with EDA, and **correlation, not causation**.

## 6. API

Model loaded **once** at startup (lifespan) with a self-check prediction; endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (independent of model) |
| `GET /ready` | readiness: artifact loaded & serving (503 otherwise) |
| `GET /version` | service + model version + estimator |
| `POST /predict` | single prediction |
| `POST /predict/batch` | ≤ 64 items, vectorized (~2 ms/item measured), per-item error attribution fallback |
| `GET /docs` | OpenAPI docs |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "C-777",
    "features": {
      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
      "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
      "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
      "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
      "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": null
    }
  }'
```

Response:

```json
{
  "customer_id": "C-777",
  "prediction": "No",
  "churn_probability": 0.4849,
  "applied_threshold": 0.56,
  "model_version": "1.0.0",
  "request_id": "dcffea49b732448db6f5c596a2cc3c0c"
}
```

**Hardening (all live-tested)**: strict pydantic contract (`extra=forbid`, literal
categories, bounded numerics) → 422 with actionable details; structural pandera check at
the predictor; `OneHotEncoder(handle_unknown="ignore")` absorbs unseen categories in
batch jobs; 256 KiB body cap → 413; fixed-window per-IP rate limit → 429 with
`Retry-After`; optional bearer auth (set `API_AUTH_TOKEN`); restrictive CORS (disabled
unless configured); security headers (`nosniff`, `DENY`, `no-store`); request-id
propagation; JSON structured logs with latency per request — request bodies and raw
customer identifiers are never logged (ids are SHA-256-truncated).

## 7. Monitoring

* **Prediction log** — every served prediction appended to
  `data/prediction_logs/predictions.jsonl` (features, score, version, latency).
* **Drift (PSI)** — `python -m src.monitoring.drift --window 500` compares recent logs
  against training-time reference distributions (`models/reference_stats.json`) for all
  19 features. Statuses: `ok < 0.10 ≤ warning < 0.20 ≤ alert`. Exit code 1 on any alert
  → CI/cron-friendly. Verified live: it correctly flagged a skewed scoring batch.
* **System metrics** — latency/error/request counts in structured logs (ship to your
  log stack); `/health` + `/ready` for orchestrators.
* **Retraining triggers** (policy): PSI alert on ≥ 3 features, weekly churn-concept
  check when ground truth lands (recall on the labeled cohort), or scheduled quarterly
  retrain. New artifacts are produced by the same pipeline; `models/latest.json` is the
  only switch, so rollback = point it at the previous artifact.

**Escalation ladder**: warning → dashboard review; alert → page on-call, compare recent
upstream changes; performance degradation with ground truth → retrain via CI gates below.

## 8. Local quickstart

```bash
cd telco-churn-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

make data          # download + checksum-verify dataset
make eda           # EDA report + figures
make train         # baselines + Optuna tuning + eval + artifact (~8 min)
pytest             # 48 tests, incl. production-artifact regression gate
make serve         # uvicorn on :8000 (or: uvicorn src.api.main:app)
make drift         # PSI drift report over recent prediction logs
```

## 9. Docker

```bash
make docker-build                 # or: docker build -t telco-churn-api:latest .
docker run --rm -p 8000:8000 telco-churn-api:latest
# or:
docker compose up --build
```

The image is `python:3.13-slim`, runs as a **non-root** user, ships pinned dependencies
and the trained artifact, and has a `HEALTHCHECK` wired to `/health`. A new model version
= rebuilt image gated by CI tests.

## 10. CI/CD

`.github/workflows/ci.yml`: ruff lint → unit/integration tests **including dataset
ingestion and a full training run**, where the regression gate
(`tests/test_model.py::test_production_artifact_meets_performance_gate`) refuses
artifacts below F1 0.52 / ROC-AUC 0.80 → container build. No model reaches deployment
without passing the gates; mypy runs advisory until typing coverage completes.

## 11. Configuration & environment

Non-secret config: `configs/config.yaml` (split sizes, CV folds, trial budgets, metric
gates, rate limits, PSI thresholds). Overrides via env: `CONFIG_PATH`, `RAW_DATA_PATH`,
`MODEL_DIR`, `PREDICTION_LOG_PATH`, `LOG_LEVEL`, `API_HOST`, `API_PORT`,
`MODEL_ARTIFACT_PATH`. Secrets: `API_AUTH_TOKEN`, `API_CORS_ORIGINS` (see
`.env.example`). Nothing secret ever lives in files.

## 12. Reproducibility

Fixed seeds end-to-end (`random_state: 42` for splits, CV, TPE sampler, estimators);
dataset pinned by SHA-256; artifact metadata records python/library versions, git commit,
config, dataset hash, training timestamp and full metrics (see
`models/churn_model_1.0.0_*.meta.json`). Two consecutive full training runs in this
workspace produced **identical** metrics, confirming determinism.

## 13. Verified vs not verifiable in this environment

Verified here by execution: ingestion checksum, EDA, full training + tuning, hold-out
evaluation, leakage audit, 48/48 tests, live API smoke tests (valid/invalid/batch/auth/
rate-limit/size-cap), drift monitor on logged predictions, deterministic re-run.

**Not verifiable here**: actual Docker builds (no docker daemon in this sandbox) and the
GitHub Actions pipeline (hosted runners) — configs are complete and standard but should
be exercised on a machine with Docker/CI access before relying on them.

## 14. Known limitations & next steps

* Snapshot data: no time dimension, so *when* a customer churns and cohort effects are
  out of scope; a temporal dataset would switch the design to TimeSeriesSplit/GroupKFold.
* Single operator, synthetic-sample provenance (IBM demo data): recalibrate thresholds
  and re-audit fairness slices on your real population before production use.
* Improvement over tuned logistic regression is small (+0.0095 F1): if operational
  simplicity wins in review, `configs/config.yaml → model.candidates: ["logreg"]`
  reproduces a fully supported simpler system.
* Fairness slices (gender, senior-citizen) are reported observationally in
  `reports/evaluation/metrics.json`; extend before using scores for automated decisions.
* Next steps: SHAP local explanations for retention agents, ONNX export for
  sub-10-ms p99, champion/challenger shadow deployment, ground-truth loop for
  concept-drift metrics.
