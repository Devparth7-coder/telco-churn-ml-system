"""Baseline models.

Two baselines anchor the evaluation:

* ``majority`` — always predicts the majority class. Any useful model must
  beat its F1 on the churn class (trivially 0.0 when predicting "No" always).
* ``logreg_default`` — L2 logistic regression with default regularization and
  class-balanced weights, run through identical preprocessing and CV as the
  candidates so comparisons are apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_full_pipeline, build_preprocessor


@dataclass(frozen=True)
class BaselineResult:
    name: str
    f1: float
    recall: float
    precision: float
    roc_auc: float


def majority_baseline(y_train: pd.Series, y_eval: np.ndarray) -> BaselineResult:
    """Evaluate the always-predict-majority-class strategy."""
    majority = int(y_train.mode().iloc[0]) if hasattr(y_train, "mode") else int(np.bincount(y_train).argmax())
    preds = np.full(len(y_eval), majority, dtype=int)
    tp = int(((preds == 1) & (y_eval == 1)).sum())
    fp = int(((preds == 1) & (y_eval == 0)).sum())
    fn = int(((preds == 0) & (y_eval == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BaselineResult(name="majority", f1=f1, recall=recall, precision=precision, roc_auc=0.5)


def logreg_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, random_state: int
) -> tuple[BaselineResult, Pipeline]:
    """Cross-validated logistic-regression baseline on the training split only."""
    estimator = LogisticRegression(class_weight="balanced", max_iter=2000)
    pipeline = build_full_pipeline(build_preprocessor(), estimator)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    y_prob = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y = y_train.to_numpy()

    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    from sklearn.metrics import roc_auc_score

    return (
        BaselineResult(name="logreg_default", f1=f1, recall=recall, precision=precision, roc_auc=float(roc_auc_score(y, y_prob))),
        pipeline,
    )
