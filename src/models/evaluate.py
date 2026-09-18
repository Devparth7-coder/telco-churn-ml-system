"""Evaluation metrics, threshold selection, plots, and reporting.

The primary metric is **F1 on the churn (positive) class**: the retention team
pays a cost for every unnecessary retention offer (false positive) and loses
revenue for every missed churner (false negative), so neither precision nor
recall alone is appropriate; F1 balances them while staying interpretable.
ROC-AUC and PR-AUC are reported as threshold-free secondary metrics, and
Brier score tracks probability calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute the full metric suite at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "churn_rate": float(y_true.mean()),
        "accuracy": accuracy,
        "precision_churn": precision,
        "recall_churn": recall,
        "f1": f1,
        "f1_macro": float(classification_report(y_true, y_pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if len(np.unique(y_true)) > 1:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        # sklearn returns recall in decreasing order; integrate ascending.
        order = np.argsort(recall_arr)
        metrics["pr_auc"] = float(np.trapezoid(precision_arr[order], recall_arr[order]))
    else:
        metrics["pr_auc"] = float("nan")
    return metrics


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, low: float = 0.10, high: float = 0.90, steps: int = 81
) -> tuple[float, float]:
    """Pick the threshold maximising churn-class F1 on the given split."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(low, high, steps):
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        if metrics["f1"] > best_f1:
            best_threshold, best_f1 = float(threshold), metrics["f1"]
    return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Plots — every figure answers a specific evaluation question.
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    """Q: how well does the model separate churners across all thresholds?"""
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_true, y_prob), estimator_name="final").plot(ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    """Q: precision/recall trade-off under the churn class imbalance?"""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name="final")
    ax.set_title("Precision-Recall curve (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    """Q: are predicted probabilities trustworthy as risk scores?"""
    fig, ax = plt.subplots(figsize=(6, 5))
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    ax.plot(mean_predicted, fraction_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path) -> Path:
    """Q: where exactly do errors fall at the deployed threshold?"""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred No", "pred Yes"])
    ax.set_yticks([0, 1], ["true No", "true Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"Confusion matrix @ threshold={threshold:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    """Q: how separable are the score distributions of the two classes?"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="retained", density=True)
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="churned", density=True)
    ax.set_xlabel("P(churn)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_evaluation_figures(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, figures_dir: Path
) -> dict[str, str]:
    """Render all evaluation figures; returns a name->path mapping."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "roc_curve": str(plot_roc_curve(y_true, y_prob, figures_dir / "roc_curve.png")),
        "pr_curve": str(plot_pr_curve(y_true, y_prob, figures_dir / "pr_curve.png")),
        "calibration": str(plot_calibration(y_true, y_prob, figures_dir / "calibration.png")),
        "confusion_matrix": str(plot_confusion(y_true, y_prob, threshold, figures_dir / "confusion_matrix.png")),
        "score_distribution": str(
            plot_probability_distribution(y_true, y_prob, figures_dir / "score_distribution.png")
        ),
    }
