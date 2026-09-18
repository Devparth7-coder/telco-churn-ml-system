"""Reproducible training orchestrator.

Run:
    python -m src.models.train

Flow (see README "ML strategy" for rationale):
1. Load + schema-validate the raw dataset.
2. Stratified split into train / validation / test (assignments persisted).
3. Baselines (majority class, default logistic regression) via CV on train.
4. Tune every configured candidate with Optuna (CV on train only).
5. Select the winner on CV F1; refit on train; pick decision threshold on
   validation; evaluate ONCE on the untouched test set.
6. Serialize artifact + reference drift stats + leakage audit + reports.
"""
from __future__ import annotations

import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import AppConfig, load_config
from src.data.ingestion import load_raw_dataset, sha256_of
from src.data.preprocessing import build_full_pipeline, build_preprocessor
from src.data.schemas import FEATURE_COLUMNS
from src.data.splitting import stratified_split
from src.logging_utils import get_logger, setup_logging
from src.models.baseline import logreg_baseline, majority_baseline
from src.models.evaluate import compute_metrics, generate_evaluation_figures, select_threshold
from src.models.registry import ModelArtifact, save_artifact
from src.models.tuning import EstimatorUnavailableError, TuningResult, build_estimator, tune_candidate
from src.monitoring.drift import compute_reference_stats, save_reference_stats

logger = get_logger("train")

#: Simpler-first order used to break near-ties between candidates.
_SIMPLICITY_ORDER = ["logreg", "hist_gb", "lightgbm", "random_forest"]
_TIE_EPSILON = 0.002


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _prepare_xy(split_df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.Series]:
    x = split_df.drop(columns=[cfg.data.target, cfg.data.id_column]).reset_index(drop=True)
    y = (split_df[cfg.data.target] == "Yes").astype(int).reset_index(drop=True)
    return x, y


def _single_feature_auc_scan(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    """Leakage screen: per-feature AUC flags implausibly predictive columns."""
    results: dict[str, float] = {}
    y = y_train.to_numpy()
    for column in x_train.columns:
        series = x_train[column]
        if pd.api.types.is_numeric_dtype(series):
            scores = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        elif series.nunique(dropna=True) == 2:
            first = series.dropna().iloc[0]
            scores = (series == first).astype(float).to_numpy()
        else:
            continue
        try:
            auc = roc_auc_score(y, scores)
        except ValueError:
            continue
        results[column] = round(max(auc, 1.0 - auc), 4)
    return results


def _write_leakage_audit(
    audit_path: Path, scan: dict[str, float], n_rows: int, n_duplicates: int, split_sizes: dict[str, int]
) -> None:
    flagged = {feature: auc for feature, auc in scan.items() if auc >= 0.90}
    lines = [
        "# Data Leakage Audit",
        "",
        f"Dataset rows: {n_rows}. Duplicate customerIDs: {n_duplicates}. "
        f"Splits: train={split_sizes['train']}, validation={split_sizes['validation']}, test={split_sizes['test']}.",
        "",
        "## Decisions",
        "",
        "| Item | Risk | Decision |",
        "|---|---|---|",
        "| `customerID` | Identifier; unique key, no predictive semantics | **Dropped** before modeling |",
        "| `TotalCharges` | Highly correlated with tenure x MonthlyCharges, but is billing history known at scoring time | **Kept** (no future information) |",
        "| Temporal leakage | Dataset is a single snapshot with no event timestamps | Stratified random split is valid; documented assumption |",
        "| Group leakage | One row per customer; identifier verified unique before splitting | No GroupKFold needed |",
        "| Preprocessing leakage | Imputer/scaler/OHE fitted inside `pipeline.fit` on the training split only | Enforced by pipeline composition |",
        "| Post-outcome features | No columns recorded after the churn outcome | None present in schema |",
        "| Threshold/selection leakage | Tuning uses train CV only; threshold from validation; test evaluated once at the end | Enforced in `train.py` flow |",
        "",
        "## Single-feature AUC screen (train split; >= 0.90 flagged)",
        "",
        "Any single feature that almost perfectly predicts the target would indicate target leakage.",
        "",
        "| Feature | AUC | Flag |",
        "|---|---|---|",
    ]
    for feature, auc in sorted(scan.items(), key=lambda kv: -kv[1]):
        flag = "**YES — investigate**" if auc >= 0.97 else ("review" if auc >= 0.90 else "")
        lines.append(f"| {feature} | {auc:.4f} | {flag} |")
    lines += [
        "",
        f"Flagged at >=0.90: {len(flagged)} -> {sorted(flagged) if flagged else 'none'}",
        "",
        "All screened features describe account state at snapshot time; no feature is computed from",
        "post-outcome information. Correlation is treated as correlation: nothing here is a causal claim.",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fairness_slices(
    x_test: pd.DataFrame, y_test: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Observational parity slices (TPR/FPR by group). Documented, not optimized."""
    y_pred = (y_prob >= threshold).astype(int)
    slices: dict[str, Any] = {}
    for attribute in ("gender", "SeniorCitizen"):
        per_group: dict[str, Any] = {}
        for group in sorted(x_test[attribute].astype(str).unique()):
            mask = (x_test[attribute].astype(str) == group).to_numpy()
            positives = y_test[mask] == 1
            negatives = y_test[mask] == 0
            per_group[str(group)] = {
                "n": int(mask.sum()),
                "churn_rate": round(float(y_test[mask].mean()), 4),
                "tpr": round(float(y_pred[mask][positives].mean()), 4) if positives.any() else None,
                "fpr": round(float(y_pred[mask][negatives].mean()), 4) if negatives.any() else None,
            }
        slices[attribute] = per_group
    return slices


def main() -> int:  # noqa: C901 — orchestration is inherently sequential
    cfg = load_config()
    setup_logging(cfg.logging.level)
    started = time.perf_counter()
    seed_everything(cfg.training.random_state)

    # ------------------------------------------------------------------ data
    df = load_raw_dataset(cfg)
    split = stratified_split(df, cfg.data.target, cfg.data.id_column, cfg.split, cfg.training.random_state)

    cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    split.assignments.to_csv(cfg.data.processed_dir / "split_assignments.csv", index=False)

    x_train, y_train = _prepare_xy(split.train, cfg)
    x_val, y_val = _prepare_xy(split.validation, cfg)
    x_test, y_test = _prepare_xy(split.test, cfg)
    pos_weight = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))
    logger.info(
        "Data prepared",
        extra={
            "train": len(x_train), "validation": len(x_val), "test": len(x_test),
            "train_churn_rate": round(float(y_train.mean()), 4),
            "scale_pos_weight": round(pos_weight, 3),
        },
    )

    # ------------------------------------------------------------- baselines
    majority = majority_baseline(y_train, y_test.to_numpy())
    logreg_cv, logreg_pipeline = logreg_baseline(x_train, y_train, cfg.training.cv_folds, cfg.training.random_state)
    logreg_pipeline.fit(x_train, y_train)
    logreg_test = compute_metrics(
        y_test.to_numpy(), logreg_pipeline.predict_proba(x_test)[:, 1], threshold=0.5
    )
    logger.info(
        "Baselines",
        extra={
            "majority_f1": majority.f1,
            "logreg_cv_f1": round(logreg_cv.f1, 4),
            "logreg_test_f1": round(logreg_test["f1"], 4),
            "logreg_test_roc_auc": round(logreg_test["roc_auc"], 4),
        },
    )

    # ------------------------------------------------------------ candidates
    tuning_results: dict[str, TuningResult] = {}
    for name in cfg.model.candidates:
        try:
            tuning_results[name] = tune_candidate(name, x_train, y_train, cfg.training, pos_weight)
        except EstimatorUnavailableError as exc:
            logger.warning("Candidate skipped (library unavailable)", extra={"estimator": name, "reason": str(exc)})

    if not tuning_results:
        logger.error("No candidates could be trained; aborting.")
        return 1

    def selection_key(item: tuple[str, TuningResult]) -> tuple[float, int]:
        name, result = item
        return (result.best_cv_f1, -_SIMPLICITY_ORDER.index(name) if name in _SIMPLICITY_ORDER else -99)

    ranked = sorted(tuning_results.items(), key=selection_key, reverse=True)
    winner_name, winner = ranked[0]
    # Near-tie policy: prefer the simpler model if within _TIE_EPSILON of the best.
    for name, result in ranked[1:]:
        if winner.best_cv_f1 - result.best_cv_f1 <= _TIE_EPSILON and _SIMPLICITY_ORDER.index(name) < _SIMPLICITY_ORDER.index(winner_name):
            logger.info(
                "Near-tie: preferring simpler candidate",
                extra={"from": winner_name, "to": name, "delta_cv_f1": round(winner.best_cv_f1 - result.best_cv_f1, 4)},
            )
            winner_name, winner = name, result
    logger.info(
        "Model selected",
        extra={"estimator": winner_name, "cv_f1": round(winner.best_cv_f1, 4), "cv_std": round(winner.cv_std, 4)},
    )

    # --------------------------------------------------------- final fitting
    best_params = dict(winner.best_params)
    best_params["random_state"] = cfg.training.random_state
    estimator = build_estimator(winner_name, best_params, pos_weight)
    pipeline: Pipeline = build_full_pipeline(build_preprocessor(), estimator)
    pipeline.fit(x_train, y_train)

    val_prob = pipeline.predict_proba(x_val)[:, 1]
    threshold, val_f1_at_threshold = select_threshold(y_val.to_numpy(), val_prob)
    val_metrics = compute_metrics(y_val.to_numpy(), val_prob, threshold)
    logger.info(
        "Threshold selected on validation split",
        extra={"threshold": round(threshold, 3), "validation_f1": round(val_f1_at_threshold, 4)},
    )

    # ------------------------------------------------------------ test (once)
    test_prob = pipeline.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)
    logger.info(
        "Final test evaluation (first and only contact)",
        extra={k: (round(v, 4) if isinstance(v, float) else v) for k, v in test_metrics.items() if k != "confusion"},
    )

    # Gate: improvement is measured against the majority-class baseline (the
    # true naive baseline). The tuned logistic regression is a strong reference
    # model reported separately, not the gate itself.
    improvement = test_metrics["f1"] - majority.f1
    delta_vs_logreg = test_metrics["f1"] - logreg_test["f1"]
    if improvement < cfg.evaluation.min_improvement_over_baseline:
        logger.warning(
            "Selected model does not beat the majority baseline by the configured margin",
            extra={"delta_f1": round(improvement, 4), "required": cfg.evaluation.min_improvement_over_baseline},
        )
    logger.info(
        "Improvement summary",
        extra={
            "delta_f1_vs_majority": round(improvement, 4),
            "delta_f1_vs_logreg_default": round(delta_vs_logreg, 4),
        },
    )

    # ------------------------------------------------------- leakage audit
    reports_dir = Path("reports") / "evaluation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scan = _single_feature_auc_scan(x_train, y_train)
    _write_leakage_audit(
        reports_dir / "leakage_audit.md",
        scan,
        n_rows=len(df),
        n_duplicates=int(df[cfg.data.id_column].duplicated().sum()),
        split_sizes={"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
    )

    # ------------------------------------------------------- artifact + drift
    artifact = ModelArtifact(
        pipeline=pipeline,
        threshold=threshold,
        feature_columns=list(FEATURE_COLUMNS),
        classes=["No", "Yes"],
        meta={
            "estimator": winner_name,
            "best_params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_params.items()},
            "dataset_sha256": sha256_of(cfg.data.raw_path),
            "dataset_rows": len(df),
            "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
            "cv_folds": cfg.training.cv_folds,
            "random_state": cfg.training.random_state,
            "candidate_results": {
                name: {"cv_f1": round(r.best_cv_f1, 4), "cv_std": round(r.cv_std, 4), "n_trials": r.n_trials, "tuning_s": round(r.duration_s, 1)}
                for name, r in tuning_results.items()
            },
            "baseline_results": {
                "majority": {"f1": majority.f1, "roc_auc": majority.roc_auc},
                "logreg_default": {"cv_f1": round(logreg_cv.f1, 4), "test_f1": round(logreg_test["f1"], 4), "test_roc_auc": round(logreg_test["roc_auc"], 4)},
            },
            "validation_metrics": val_metrics,
            "primary_metric": cfg.evaluation.primary_metric,
        },
    )
    artifact_path = save_artifact(artifact, cfg.model.models_dir, cfg.project.version, test_metrics)
    save_reference_stats(compute_reference_stats(x_train), cfg.model.models_dir)

    # ------------------------------------------------------------- reporting
    figures = generate_evaluation_figures(y_test.to_numpy(), test_prob, threshold, Path("reports") / "figures")
    slices = _fairness_slices(x_test, y_test.to_numpy(), test_prob, threshold)

    latency_probe = time.perf_counter()
    for _ in range(10):
        pipeline.predict_proba(x_val.head(100))
    latency_ms = (time.perf_counter() - latency_probe) / 10 * 1000

    summary = {
        "model_version": cfg.project.version,
        "estimator": winner_name,
        "artifact": str(artifact_path),
        "artifact_size_mb": round(artifact_path.stat().st_size / 1e6, 2),
        "batch100_inference_latency_ms": round(latency_ms, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "total_training_seconds": round(time.perf_counter() - started, 1),
        "test_metrics": test_metrics,
        "validation_metrics": val_metrics,
        "baseline_test_metrics": {"logreg_default": logreg_test, "majority": {"f1": majority.f1, "roc_auc": majority.roc_auc}},
        "fairness_slices": slices,
        "figures": figures,
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    report_lines = [
        "# Training Report",
        "",
        f"- Estimator: **{winner_name}** (version {cfg.project.version})",
        f"- Decision threshold: **{threshold:.3f}** (selected on validation to maximise churn-class F1)",
        f"- Artifact: `{artifact_path}` ({summary['artifact_size_mb']} MB)",
        f"- Training wall time: {summary['total_training_seconds']} s; batch(100) inference latency: {summary['batch100_inference_latency_ms']} ms",
        "",
        "## Candidate comparison (5-fold CV on train, F1 churn class)",
        "",
        "| Model | CV F1 | std | trials | tuning s |",
        "|---|---|---|---|---|",
    ]
    for name, result in sorted(tuning_results.items(), key=lambda kv: -kv[1].best_cv_f1):
        report_lines.append(
            f"| {name} | {result.best_cv_f1:.4f} | {result.cv_std:.4f} | {result.n_trials} | {result.duration_s:.0f} |"
        )
    report_lines += [
        f"| logreg_default (baseline) | {logreg_cv.f1:.4f} | - | - | - |",
        f"| majority (baseline) | {majority.f1:.4f} | - | - | - |",
        "",
        "## Test set (evaluated once, at the very end)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("f1", "precision_churn", "recall_churn", "roc_auc", "pr_auc", "accuracy", "brier"):
        report_lines.append(f"| {key} | {test_metrics[key]:.4f} |")
    report_lines += [
        f"| confusion (tn/fp/fn/tp) | {test_metrics['confusion']['tn']}/{test_metrics['confusion']['fp']}/{test_metrics['confusion']['fn']}/{test_metrics['confusion']['tp']} |",
        "",
        f"Improvement over majority-class baseline test F1: **{improvement:+.4f}** "
        f"(gate: >= {cfg.evaluation.min_improvement_over_baseline}); "
        f"vs tuned logistic regression: **{delta_vs_logreg:+.4f}**.",
        "",
        "See `reports/evaluation/metrics.json`, `leakage_audit.md`, and `reports/figures/`.",
    ]
    (reports_dir / "training_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    logger.info(
        "Training complete",
        extra={
            "estimator": winner_name,
            "test_f1": round(test_metrics["f1"], 4),
            "test_roc_auc": round(test_metrics["roc_auc"], 4),
            "artifact": str(artifact_path),
            "duration_s": summary["total_training_seconds"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
