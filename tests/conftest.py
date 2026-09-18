"""Shared fixtures.

The ``trained_artifact`` fixture trains a *compact* model (small subsample, few
boosting rounds) so the whole suite runs in seconds while still exercising the
real serialization/inference code paths. Integration tests against the full
production artifact run conditionally (see ``tests/test_model.py``).
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import SMOKE_FEATURES  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.ingestion import load_raw_dataset  # noqa: E402
from src.data.preprocessing import build_full_pipeline, build_preprocessor  # noqa: E402
from src.data.schemas import FEATURE_COLUMNS  # noqa: E402
from src.models.evaluate import select_threshold  # noqa: E402
from src.models.registry import ModelArtifact, save_artifact  # noqa: E402
from src.models.tuning import build_estimator  # noqa: E402

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

requires_dataset = pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not downloaded")


@dataclass
class SmallSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_hold: pd.DataFrame
    y_hold: pd.Series


def make_payload(**overrides) -> dict:
    """A valid /predict payload; kwargs override individual feature values."""
    features = copy.deepcopy(SMOKE_FEATURES)
    features.update(overrides)
    return {"customer_id": "TEST-0001", "features": features}


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
@requires_dataset
def raw_data(cfg):
    return load_raw_dataset(cfg)


@pytest.fixture(scope="session")
@requires_dataset
def small_split(raw_data) -> SmallSplit:
    sample = raw_data.sample(n=1800, random_state=7)
    train_df, hold_df = train_test_split(sample, test_size=0.3, stratify=sample["Churn"], random_state=7)

    def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x = frame.drop(columns=["Churn", "customerID"]).reset_index(drop=True)
        y = (frame["Churn"] == "Yes").astype(int).reset_index(drop=True)
        return x, y

    x_train, y_train = xy(train_df)
    x_hold, y_hold = xy(hold_df)
    return SmallSplit(x_train=x_train, y_train=y_train, x_hold=x_hold, y_hold=y_hold)


@dataclass
class TrainedArtifact:
    path: Path
    models_dir: Path
    threshold: float


@pytest.fixture(scope="session")
@requires_dataset
def trained_artifact(small_split, tmp_path_factory) -> TrainedArtifact:
    """Train a compact HistGB model and serialize it like production does."""
    pos_weight = float((small_split.y_train == 0).sum() / max(int((small_split.y_train == 1).sum()), 1))
    estimator = build_estimator(
        "hist_gb",
        {"learning_rate": 0.1, "max_iter": 80, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1, "random_state": 42},
        pos_weight,
    )
    pipeline = build_full_pipeline(build_preprocessor(), estimator)
    pipeline.fit(small_split.x_train, small_split.y_train)

    hold_prob = pipeline.predict_proba(small_split.x_hold)[:, 1]
    threshold, _ = select_threshold(small_split.y_hold.to_numpy(), hold_prob)

    artifact = ModelArtifact(
        pipeline=pipeline,
        threshold=threshold,
        feature_columns=list(FEATURE_COLUMNS),
        classes=["No", "Yes"],
        meta={"estimator": "hist_gb", "model_version": "test-fixture"},
    )
    models_dir = tmp_path_factory.mktemp("models")
    path = save_artifact(artifact, models_dir, "test-fixture", metrics={"f1": 0.0})
    return TrainedArtifact(path=path, models_dir=models_dir, threshold=threshold)
