"""Feature engineering.

All derived features are computed **only from information that exists at
inference time** (account snapshot fields) — no post-outcome or aggregate
future data is used, per the leakage audit.

Derived features
----------------
* ``tenure_group``: binned tenure (lifecycle stage).
* ``service_count``: number of active services (phone + internet + 6 add-ons).
* ``avg_monthly_charges``: ``TotalCharges / (tenure + 1)`` — the +1 guards the
  tenure=0 edge case; NaN TotalCharges stays NaN and is imputed downstream.
* ``no_security_support``: flag for internet customers without OnlineSecurity
  or TechSupport, a known churn-risk profile in telecom data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

DERIVED_NUMERIC_FEATURES = ["service_count", "avg_monthly_charges", "no_security_support"]
DERIVED_CATEGORICAL_FEATURES = ["tenure_group"]
#: Creation order in transform() — get_feature_names_out must match exactly.
DERIVED_FEATURES_IN_CREATION_ORDER = [
    "tenure_group",
    "service_count",
    "avg_monthly_charges",
    "no_security_support",
]

_TENURE_BINS = [-0.001, 12, 24, 48, np.inf]
_TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless deterministic feature transforms (fit is a no-op).

    Being stateless is deliberate: there are no fitted statistics that could
    leak between splits, and inference behaviour is identical to training.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        self.feature_names_in_ = np.asarray(list(X.columns))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        out["tenure_group"] = pd.cut(
            out["tenure"].astype(float), bins=_TENURE_BINS, labels=_TENURE_LABELS
        ).astype(str)

        addon_count = (out[ADDON_SERVICES] == "Yes").sum(axis=1)
        out["service_count"] = (
            addon_count
            + (out["PhoneService"] == "Yes").astype(int)
            + (out["InternetService"] != "No").astype(int)
        ).astype("int64")

        tenure = out["tenure"].astype(float)
        total = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["avg_monthly_charges"] = total / (tenure + 1.0)

        out["no_security_support"] = (
            (out["InternetService"] != "No")
            & (out["OnlineSecurity"] != "Yes")
            & (out["TechSupport"] != "Yes")
        ).astype("int64")

        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + DERIVED_FEATURES_IN_CREATION_ORDER)
