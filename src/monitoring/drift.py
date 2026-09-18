"""Feature drift monitoring via Population Stability Index (PSI).

At training time we snapshot reference distributions of the *raw* input
features (the ones logged at inference). The CLI then compares the most recent
prediction-log records against that reference:

* ``PSI < 0.10``  -> stable
* ``0.10 <= PSI < 0.20`` -> warning (investigate)
* ``PSI >= 0.20`` -> alert (trigger retraining investigation)

Run:
    python -m src.monitoring.drift [--window 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.logging_utils import get_logger, setup_logging

logger = get_logger("drift")

REFERENCE_STATS_FILENAME = "reference_stats.json"
_EPS = 1e-6
_N_QUANTILE_BINS = 10

#: Feature kinds used from the raw (logged) feature contract.
NUMERIC_REFERENCE_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_REFERENCE_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def compute_reference_stats(raw_features: pd.DataFrame) -> dict:
    """Snapshot training-time feature distributions for later PSI comparison."""
    stats: dict = {"numeric": {}, "categorical": {}}

    for feature in NUMERIC_REFERENCE_FEATURES:
        values = pd.to_numeric(raw_features[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        quantiles = np.quantile(values, np.linspace(0, 1, _N_QUANTILE_BINS + 1))
        edges = np.unique(quantiles)
        counts, edges = np.histogram(values, bins=edges)
        props = counts / max(counts.sum(), 1)
        stats["numeric"][feature] = {
            "edges": edges.tolist(),
            "proportions": props.tolist(),
        }

    for feature in CATEGORICAL_REFERENCE_FEATURES:
        counts = raw_features[feature].astype(str).value_counts(normalize=True)
        stats["categorical"][feature] = counts.to_dict()

    return stats


def save_reference_stats(stats: dict, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / REFERENCE_STATS_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    return path


def load_reference_stats(models_dir: Path) -> dict:
    path = models_dir / REFERENCE_STATS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No reference stats at {path}; run training first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    expected = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift(reference: dict, new_features: pd.DataFrame, warning: float, alert: float) -> dict:
    """Compute per-feature PSI of ``new_features`` against the reference."""
    report: dict[str, dict] = {}

    for feature, ref in reference.get("numeric", {}).items():
        if feature not in new_features:
            continue
        edges = np.asarray(ref["edges"], dtype=float)
        values = pd.to_numeric(new_features[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0 or len(edges) < 2:
            continue
        counts, _ = np.histogram(values, bins=edges)
        actual = counts / max(counts.sum(), 1)
        value = psi(np.asarray(ref["proportions"], dtype=float), actual)
        report[feature] = _classify(value, warning, alert)

    for feature, ref_props in reference.get("categorical", {}).items():
        if feature not in new_features:
            continue
        categories = sorted(ref_props.keys())
        observed_counts = new_features[feature].astype(str).value_counts()
        expected = np.array([ref_props.get(c, 0.0) for c in categories] + [0.0])
        actual = np.array([observed_counts.get(c, 0) for c in categories] + [
            int(observed_counts.drop(index=[c for c in observed_counts.index if c in categories], errors="ignore").sum())
        ], dtype=float)
        if actual.sum() == 0:
            continue
        # Fold unseen-at-training categories into an "__other__" bucket.
        value = psi(expected, actual)
        report[feature] = _classify(value, warning, alert)

    return report


def _classify(value: float, warning: float, alert: float) -> dict:
    status = "ok" if value < warning else ("warning" if value < alert else "alert")
    return {"psi": round(value, 4), "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute feature drift (PSI) for recent predictions.")
    parser.add_argument("--window", type=int, default=500, help="How many recent log records to use.")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.logging.level)

    from src.monitoring.prediction_log import PredictionLogger

    records = PredictionLogger(cfg.monitoring.prediction_log_path).read_recent(args.window)
    if len(records) < cfg.monitoring.drift_min_samples:
        logger.warning(
            "Not enough logged predictions for drift analysis",
            extra={"records": len(records), "minimum": cfg.monitoring.drift_min_samples},
        )
        return 0

    features = pd.DataFrame([record.get("features", {}) for record in records])
    reference = load_reference_stats(cfg.model.models_dir)
    report = compute_drift(reference, features, cfg.monitoring.psi_warning, cfg.monitoring.psi_alert)

    worst = max((item["psi"] for item in report.values()), default=0.0)
    statuses: dict[str, int] = {
        status: sum(1 for item in report.values() if item["status"] == status)
        for status in ("ok", "warning", "alert")
    }
    summary = {
        "n_records": len(records),
        "features_checked": len(report),
        "worst_psi": worst,
        "statuses": statuses,
        "features": report,
    }
    reports_dir = Path("reports") / "evaluation" if Path("reports").exists() else Path(".")
    out_path = reports_dir / "drift_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Drift report", extra={"worst_psi": worst, "statuses": statuses, "report": str(out_path)})
    for feature, item in sorted(report.items(), key=lambda kv: -kv[1]["psi"])[:10]:
        logger.info("PSI", extra={"feature": feature, **item})
    return 1 if statuses["alert"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
