"""Robustness tests: the model and API must degrade gracefully, never crash."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.schemas import SchemaValidationError
from src.models.predict import Predictor
from tests.conftest import requires_dataset

pytestmark = requires_dataset


@pytest.fixture(scope="module")
def predictor(trained_artifact) -> Predictor:
    return Predictor.from_path(trained_artifact.path)


def test_extreme_but_schema_valid_values_still_score(predictor, small_split):
    extreme = small_split.x_hold.iloc[[0]].copy()
    extreme["tenure"] = 120
    extreme["MonthlyCharges"] = 999.0
    extreme["TotalCharges"] = 199_000.0
    result = predictor.predict_dataframe(extreme)
    probability = result["churn_probability"].iloc[0]
    assert np.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_zero_tenure_new_customer_scores(predictor, small_split):
    new_customer = small_split.x_hold.iloc[[0]].copy()
    new_customer["tenure"] = 0
    new_customer["TotalCharges"] = None  # mirrors the upstream blank-cell quirk
    result = predictor.predict_dataframe(new_customer)
    assert np.isfinite(result["churn_probability"].iloc[0])


def test_all_missing_optionals_do_not_crash(predictor, small_split):
    frame = small_split.x_hold.iloc[[0]].copy()
    frame["TotalCharges"] = None
    frame["OnlineSecurity"] = "No internet service"
    result = predictor.predict_dataframe(frame)
    assert result["prediction"].iloc[0] in ("Yes", "No")


def test_unseen_category_is_ignored_not_fatal(predictor, small_split):
    frame = small_split.x_hold.iloc[[0]].copy()
    frame["PaymentMethod"] = "Cryptocurrency"
    result = predictor.predict_dataframe(frame)
    assert np.isfinite(result["churn_probability"].iloc[0])


def test_invalid_domain_values_raise_validation_error(predictor, small_split):
    frame = small_split.x_hold.iloc[[0]].copy()
    frame["MonthlyCharges"] = -10.0
    with pytest.raises(SchemaValidationError):
        predictor.predict_dataframe(frame)


def test_large_batch_scores_in_one_call(predictor, small_split):
    result = predictor.predict_dataframe(small_split.x_hold)
    assert len(result) == len(small_split.x_hold)
    assert ((result["churn_probability"] >= 0) & (result["churn_probability"] <= 1)).all()
