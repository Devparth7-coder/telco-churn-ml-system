"""Feature engineering and preprocessing tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessor
from src.data.schemas import FEATURE_COLUMNS
from src.features.engineering import (
    DERIVED_CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    FeatureEngineer,
)
from tests.conftest import requires_dataset  # noqa: F401


def test_engineer_creates_expected_columns(small_split):
    out = FeatureEngineer().fit_transform(small_split.x_train)
    for column in DERIVED_NUMERIC_FEATURES + DERIVED_CATEGORICAL_FEATURES:
        assert column in out.columns


def test_engineer_handles_zero_tenure_and_missing_totals():
    row = pd.DataFrame(
        [
            {
                **{col: "No" for col in ["PhoneService"]},
                "tenure": 0,
                "TotalCharges": None,
                "MonthlyCharges": 20.0,
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "PhoneService": "Yes",
                **{
                    col: "No"
                    for col in [
                        "OnlineBackup",
                        "DeviceProtection",
                        "StreamingTV",
                        "StreamingMovies",
                    ]
                },
            }
        ]
    )
    out = FeatureEngineer().fit_transform(row)
    assert pd.isna(out["avg_monthly_charges"].iloc[0])  # NaN preserved, imputed later
    assert out["no_security_support"].iloc[0] == 1
    assert out["service_count"].iloc[0] == 2  # phone + DSL internet, no add-ons


@requires_dataset
def test_preprocessor_output_is_fully_numeric_and_finite(small_split):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(small_split.x_train)
    assert isinstance(transformed, np.ndarray)
    assert np.isfinite(transformed).all()
    names = preprocessor.get_feature_names_out()
    assert len(names) == transformed.shape[1] > 0


@requires_dataset
def test_preprocessor_tolerates_unseen_category(small_split):
    preprocessor = build_preprocessor().fit(small_split.x_train)
    rogue = small_split.x_hold.iloc[[0]].copy()
    rogue["gender"] = "Unspecified"
    transformed = preprocessor.transform(rogue)
    assert np.isfinite(transformed).all()


@requires_dataset
def test_inference_frame_uses_canonical_column_contract(small_split):
    assert list(small_split.x_train.columns) == FEATURE_COLUMNS
