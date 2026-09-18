"""Hyperparameter optimization with Optuna.

Strategy
--------
* Objective: mean out-of-fold **F1 on the churn class** under stratified
  K-fold CV on the *training split only* (the validation and test sets are
  never touched here). The decision threshold is fixed at 0.5 during tuning;
  threshold selection happens afterwards on the validation split, keeping the
  two choices statistically independent.
* Sampler: seeded TPE for reproducibility; median pruner for early stopping
  of unpromising trials.
* Budgets are small and explicit (see configs/config.yaml) — no grid-search
  carpet bombing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import TrainingConfig
from src.data.preprocessing import build_full_pipeline, build_preprocessor
from src.logging_utils import get_logger

logger = get_logger("tuning")


class EstimatorUnavailableError(RuntimeError):
    """Raised when a candidate's library cannot be imported."""


def build_estimator(name: str, params: dict[str, Any], scale_pos_weight: float) -> BaseEstimator:
    """Instantiate a candidate estimator by name.

    Args:
        name: One of ``logreg``, ``random_forest``, ``hist_gb``, ``lightgbm``.
        params: Hyperparameters (already tuned, or defaults).
        scale_pos_weight: Negative/positive ratio used to handle the churn
            class imbalance (applied where the estimator supports it).
    """
    if name == "logreg":
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=params.get("random_state", 42),
            n_jobs=2,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_iter=params.get("max_iter", 300),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.0),
            class_weight="balanced",
            random_state=params.get("random_state", 42),
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise EstimatorUnavailableError("lightgbm is not installed") from exc
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 1.0),
            subsample_freq=1,
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 0.0),
            scale_pos_weight=scale_pos_weight,
            random_state=params.get("random_state", 42),
            n_jobs=2,
            verbosity=-1,
        )
    raise ValueError(f"Unknown estimator: {name}")


def _suggest_params(trial: Any, name: str) -> dict[str, Any]:
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "hist_gb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 700, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"No search space defined for {name}")


def _trials_for(name: str, cfg: TrainingConfig) -> int:
    return {
        "lightgbm": cfg.n_trials_lgbm,
        "hist_gb": cfg.n_trials_hgb,
        "random_forest": cfg.n_trials_rf,
        "logreg": cfg.n_trials_lr,
    }[name]


@dataclass
class TuningResult:
    name: str
    best_params: dict[str, Any]
    best_cv_f1: float
    cv_std: float
    n_trials: int
    duration_s: float
    trial_scores: list[float] = field(default_factory=list)


def tune_candidate(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: TrainingConfig,
    scale_pos_weight: float,
) -> TuningResult:
    """Run Optuna for one candidate and return its best configuration."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    started = time.perf_counter()

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, name)
        params["random_state"] = cfg.random_state
        estimator = build_estimator(name, params, scale_pos_weight)
        pipeline: Pipeline = build_full_pipeline(build_preprocessor(), estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=_trials_for(name, cfg), show_progress_bar=False)

    duration = time.perf_counter() - started
    best_value = float(study.best_value)
    # CV std of trials that reached the best value band, for stability reporting.
    values = [float(trial.value) for trial in study.trials if trial.value is not None]
    std = float(np.std([v for v in values if v >= best_value - 0.005]) or 0.0) if values else 0.0

    logger.info(
        "Tuning complete",
        extra={
            "estimator": name,
            "best_cv_f1": round(best_value, 4),
            "n_trials": len(study.trials),
            "duration_s": round(duration, 1),
            "best_params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in study.best_params.items()},
        },
    )
    return TuningResult(
        name=name,
        best_params=dict(study.best_params),
        best_cv_f1=best_value,
        cv_std=std,
        n_trials=len(study.trials),
        duration_s=duration,
        trial_scores=values,
    )
