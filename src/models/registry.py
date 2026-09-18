"""Model registry: artifact (de)serialization with full provenance metadata.

The serialized artifact is a single joblib file containing everything needed
for inference: the fitted preprocessing+model pipeline, the decision
threshold, feature contract, and metadata (dataset checksum, config, library
versions, git commit, evaluation metrics).

Security note: joblib/pickle artifacts can execute arbitrary code on load.
Only load artifacts produced by this project's own training pipeline from the
trusted model directory.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import git_commit

_ARTIFACT_SUFFIX = ".joblib"
LATEST_POINTER = "latest.json"


def library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:  # optional dependency
        pass
    return versions


@dataclass
class ModelArtifact:
    """Complete inference bundle for one trained model version."""

    pipeline: Any  # sklearn Pipeline (preprocessing + estimator)
    threshold: float
    feature_columns: list[str]
    classes: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def save_artifact(
    artifact: ModelArtifact, models_dir: Path, version: str, metrics: dict[str, Any]
) -> Path:
    """Persist the artifact, its metadata sidecar, and the `latest` pointer."""
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact.meta.update(
        {
            "model_version": version,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "library_versions": library_versions(),
            "test_metrics": metrics,
        }
    )
    filename = f"churn_model_{version}_{timestamp}{_ARTIFACT_SUFFIX}"
    path = models_dir / filename
    joblib.dump(asdict(artifact), path, compress=3)

    sidecar = {k: v for k, v in artifact.meta.items() if k != "pipeline"}
    with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    pointer = {"path": filename, "version": version, "trained_at_utc": artifact.meta["trained_at_utc"]}
    with open(models_dir / LATEST_POINTER, "w", encoding="utf-8") as fh:
        json.dump(pointer, fh, indent=2)
    return path


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load a serialized artifact from an explicit path."""
    payload = joblib.load(path)
    return ModelArtifact(
        pipeline=payload["pipeline"],
        threshold=float(payload["threshold"]),
        feature_columns=list(payload["feature_columns"]),
        classes=list(payload["classes"]),
        meta=dict(payload.get("meta", {})),
    )


def load_latest_artifact(models_dir: Path) -> ModelArtifact:
    """Load the artifact referenced by ``models/latest.json``."""
    pointer_path = models_dir / LATEST_POINTER
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"No {pointer_path} found — train a model first: `python -m src.models.train`."
        )
    with open(pointer_path, encoding="utf-8") as fh:
        pointer = json.load(fh)
    return load_artifact(models_dir / pointer["path"])
