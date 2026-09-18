"""Model artifact, inference, and performance-gate tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.data.schemas import SchemaValidationError
from src.models.predict import Predictor
from src.models.registry import load_artifact, load_latest_artifact
from tests.conftest import requires_dataset


@pytest.fixture(scope="module")
def predictor(trained_artifact) -> Predictor:
    return Predictor.from_path(trained_artifact.path)


@requires_dataset
def test_artifact_roundtrip(trained_artifact):
    artifact = load_artifact(trained_artifact.path)
    assert artifact.pipeline is not None
    assert 0.0 < artifact.threshold < 1.0
    assert artifact.classes == ["No", "Yes"]
    assert artifact.meta["model_version"] == "test-fixture"
    latest = load_latest_artifact(trained_artifact.models_dir)
    assert latest.meta["model_version"] == "test-fixture"


@requires_dataset
def test_prediction_shapes_types_and_probability_bounds(predictor, small_split):
    result = predictor.predict_dataframe(small_split.x_hold.head(50))
    assert len(result) == 50
    assert set(result["prediction"].unique()) <= {"Yes", "No"}
    probs = result["churn_probability"].to_numpy()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


@requires_dataset
def test_predictions_are_deterministic(predictor, small_split):
    first = predictor.predict_dataframe(small_split.x_hold.head(20))
    second = predictor.predict_dataframe(small_split.x_hold.head(20))
    assert np.allclose(first["churn_probability"], second["churn_probability"])


@requires_dataset
def test_model_beats_majority_baseline(predictor, small_split):
    result = predictor.predict_dataframe(small_split.x_hold)
    tp = int(((result["prediction"] == "Yes").to_numpy() & (small_split.y_hold == 1).to_numpy()).sum())
    positives = int((small_split.y_hold == 1).sum())
    recall = tp / positives if positives else 0.0
    # Majority class ("No") scores F1=0.0 on churn; a compact fixture model
    # must still demonstrate non-trivial recall.
    assert recall > 0.25


@requires_dataset
def test_artifact_contains_preprocessing(predictor):
    step_names = [name for name, _ in predictor.pipeline.steps]
    assert step_names == ["prep", "model"]
    prep = predictor.pipeline.named_steps["prep"]
    imputer = prep.named_steps["features"].named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_ is not None  # fitted statistics travel with the artifact


@requires_dataset
def test_schema_violation_raises_not_crashes(predictor, small_split):
    rogue = small_split.x_hold.iloc[[0]].copy()
    rogue["SeniorCitizen"] = 5
    with pytest.raises(SchemaValidationError):
        predictor.predict_dataframe(rogue)


@requires_dataset
def test_empty_input_raises_value_error(predictor, small_split):
    with pytest.raises(ValueError, match="Empty input"):
        predictor.predict_dataframe(small_split.x_hold.iloc[0:0])


@requires_dataset
def test_missing_columns_raise_value_error(predictor, small_split):
    with pytest.raises(ValueError, match="Missing required feature columns"):
        predictor.predict_dataframe(small_split.x_hold.drop(columns=["Contract"]))


# ---------------------------------------------------------------------------
# Regression gate against the real production artifact (runs only when the
# model has been trained). Prevents future iterations from silently degrading.
# ---------------------------------------------------------------------------

LATEST_POINTER = PROJECT_ROOT / "models" / "latest.json"
MIN_ACCEPTABLE_TEST_F1 = 0.52
MIN_ACCEPTABLE_TEST_ROC_AUC = 0.80


@pytest.mark.integration
@requires_dataset
@pytest.mark.skipif(not LATEST_POINTER.exists(), reason="production artifact not trained yet")
def test_production_artifact_meets_performance_gate():
    artifact = load_latest_artifact(LATEST_POINTER.parent)
    metrics = artifact.meta["test_metrics"]
    assert metrics["f1"] >= MIN_ACCEPTABLE_TEST_F1, metrics
    assert metrics["roc_auc"] >= MIN_ACCEPTABLE_TEST_ROC_AUC, metrics

    # Metadata provenance must be complete.
    for key in ("model_version", "trained_at_utc", "git_commit", "library_versions", "dataset_sha256"):
        assert key in artifact.meta, f"missing provenance field: {key}"
