"""Preprocessing pipeline construction.

Everything needed at inference time is captured inside a single scikit-learn
``Pipeline``:

    CleaningTransformer -> FeatureEngineer -> ColumnTransformer(impute+encode)

The pipeline is fitted **only on training data** (see ``src/models/train.py``)
so no validation/test statistics can leak into preprocessing. The same fitted
object ships inside the serialized artifact, guaranteeing training/inference
parity by construction.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import (
    DERIVED_CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    FeatureEngineer,
)

#: Raw numeric columns (SeniorCitizen is a 0/1 int in the source data).
RAW_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

#: Raw categorical columns (the remainder of the 19 inputs).
RAW_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Coerce ingestion dtypes into model-ready dtypes.

    Stateless (``fit`` is a no-op) so it is safe and identical at inference:

    * ``TotalCharges``: numeric-or-blank string -> float64 (blank -> NaN,
      imputed downstream).
    * String columns: whitespace-stripped.
    * ``SeniorCitizen``: cast to int.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CleaningTransformer:
        import numpy as np

        self.feature_names_in_ = np.asarray(list(X.columns))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        # Whitespace-only upstream cells (" ") become NaN here and are imputed
        # by the numeric branch downstream.
        out["TotalCharges"] = pd.to_numeric(
            out["TotalCharges"].astype(str).str.strip(), errors="coerce"
        ).astype("float64")
        out["SeniorCitizen"] = out["SeniorCitizen"].astype("int64")
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].astype(str).str.strip()
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        import numpy as np

        if input_features is None:
            return self.feature_names_in_
        return np.asarray(list(input_features))


def build_preprocessor() -> Pipeline:
    """Build the unfitted preprocessing pipeline (no estimator attached)."""
    numeric_features = RAW_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
    categorical_features = RAW_CATEGORICAL_FEATURES + DERIVED_CATEGORICAL_FEATURES

    numeric_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            # handle_unknown="ignore" makes the service robust to categories
            # never seen during training instead of failing at request time.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            ("num", numeric_branch, numeric_features),
            ("cat", categorical_branch, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            ("clean", CleaningTransformer()),
            ("engineer", FeatureEngineer()),
            ("features", column_transformer),
        ]
    )


def build_full_pipeline(preprocessor: Pipeline, estimator: BaseEstimator) -> Pipeline:
    """Attach an estimator to the preprocessor, forming the full artifact."""
    return Pipeline(
        [
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )
