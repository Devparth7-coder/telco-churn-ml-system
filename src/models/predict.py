"""Inference wrapper used by the API and batch jobs.

Loads the serialized artifact once and exposes prediction over raw feature
frames. All preprocessing happens inside the artifact's pipeline, so training
and inference can never diverge.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.schemas import FEATURE_COLUMNS, validate_inference
from src.logging_utils import get_logger
from src.models.registry import ModelArtifact, load_artifact, load_latest_artifact

logger = get_logger("predict")


class Predictor:
    """Stateless wrapper around a fitted artifact."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.pipeline = artifact.pipeline
        self.threshold = artifact.threshold

    # -- construction -------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path) -> Predictor:
        return cls(load_artifact(path))

    @classmethod
    def from_latest(cls, models_dir: Path) -> Predictor:
        return cls(load_latest_artifact(models_dir))

    @classmethod
    def load_default(cls) -> Predictor:
        """Resolve the artifact location: explicit env override wins."""
        explicit = os.environ.get("MODEL_ARTIFACT_PATH")
        if explicit:
            logger.info("Loading model from MODEL_ARTIFACT_PATH", extra={"path": explicit})
            return cls.from_path(explicit)
        cfg = load_config()
        return cls.from_latest(cfg.model.models_dir)

    # -- inference -----------------------------------------------------------
    @property
    def model_version(self) -> str:
        return str(self.artifact.meta.get("model_version", "unknown"))

    def predict_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a frame of raw features.

        Raises:
            ValueError: on empty input or missing columns.
            SchemaValidationError: on dtype/domain violations.
        """
        if features.empty:
            raise ValueError("Empty input: at least one record is required.")
        missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        frame = features[FEATURE_COLUMNS].copy()
        # Normalize TotalCharges to float64 before validation: batch callers may
        # pass the raw ingestion representation (strings with whitespace-only
        # blanks for missing). to_numeric(coerce) mirrors the training-time
        # cleaning exactly: blanks/garbage become NaN and get median-imputed.
        total = frame["TotalCharges"]
        if not pd.api.types.is_float_dtype(total):
            frame["TotalCharges"] = pd.to_numeric(
                total.astype(str).str.strip(), errors="coerce"
            ).astype("float64")
        frame = validate_inference(frame)

        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        predictions = np.where(probabilities >= self.threshold, "Yes", "No")
        return pd.DataFrame(
            {
                "churn_probability": probabilities,
                "prediction": predictions,
            }
        )

    def predict_record(self, features: dict) -> dict:
        """Score a single raw feature dictionary."""
        frame = pd.DataFrame([features])
        result = self.predict_dataframe(frame)
        return {
            "prediction": str(result["prediction"].iloc[0]),
            "churn_probability": float(result["churn_probability"].iloc[0]),
        }
