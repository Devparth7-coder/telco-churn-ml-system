"""Schema validation for the Telco Customer Churn dataset.

Two schemas are defined:

* ``RAW_SCHEMA`` — the full ingested CSV, including the identifier and target.
* ``INFERENCE_SCHEMA`` — the 19 feature columns accepted at inference time
  (no identifier, no target, ``TotalCharges`` may be missing).

Validation is enforced both at ingestion (fail fast on bad upstream data) and
at every API request (fail safely with actionable errors).
"""
from __future__ import annotations

import re

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
INTERNET_DEPENDENT = ["Yes", "No", "No internet service"]

NUMERIC_TOTAL_CHARGES = re.compile(r"^\d+(\.\d+)?$")

_RAW_COLUMNS: dict[str, Column] = {
    "customerID": Column(
        str,
        checks=[Check(lambda s: s.str.len() > 0, error="customerID must not be empty")],
        unique=True,
        nullable=False,
    ),
    "gender": Column(str, Check.isin(["Male", "Female"])),
    "SeniorCitizen": Column("int64", Check.isin([0, 1]), coerce=True),
    "Partner": Column(str, Check.isin(YES_NO)),
    "Dependents": Column(str, Check.isin(YES_NO)),
    "tenure": Column("int64", Check.in_range(0, 72), coerce=True),
    "PhoneService": Column(str, Check.isin(YES_NO)),
    "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
    "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
    "OnlineSecurity": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "OnlineBackup": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "DeviceProtection": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "TechSupport": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "StreamingTV": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "StreamingMovies": Column(str, Check.isin(INTERNET_DEPENDENT)),
    "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
    "PaperlessBilling": Column(str, Check.isin(YES_NO)),
    "PaymentMethod": Column(
        str,
        Check.isin(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
    ),
    "MonthlyCharges": Column("float64", Check.in_range(0, 130), coerce=True),
    # Whitespace-only TotalCharges is a known upstream quirk (new customers,
    # tenure=0): 11 rows contain a single space character. Those are treated
    # as missing data and imputed downstream — never silently dropped.
    "TotalCharges": Column(
        str,
        checks=[
            Check(
                lambda s: (
                    lambda stripped: stripped.str.match(NUMERIC_TOTAL_CHARGES) | (stripped == "")
                )(s.fillna("").astype(str).str.strip()),
                error="TotalCharges must be numeric or blank",
            )
        ],
        nullable=True,
    ),
    "Churn": Column(str, Check.isin(YES_NO)),
}

RAW_SCHEMA = DataFrameSchema(_RAW_COLUMNS, coerce=False, strict=False)

# ---------------------------------------------------------------------------
# Inference contract — deliberately different from the raw ingestion contract.
#
# Layered validation design:
#   * API edge (pydantic): strict literals and business ranges (see
#     src/api/schemas.py). Malformed payloads never reach this layer.
#   * This schema (pandera): structural/type safety for any caller of the
#     predictor (API, batch jobs, tests). Numeric ranges are widened to the
#     pydantic bounds, TotalCharges is accepted as numeric-or-null, and
#     categorical membership is NOT enforced here — unseen categories are
#     absorbed by the OneHotEncoder(handle_unknown="ignore") so batch scoring
#     degrades gracefully instead of failing whole jobs.
# ---------------------------------------------------------------------------
_INFERENCE_COLUMNS: dict[str, Column] = {
    "gender": Column(str, nullable=False),
    "SeniorCitizen": Column("int64", Check.isin([0, 1]), coerce=True),
    "Partner": Column(str, nullable=False),
    "Dependents": Column(str, nullable=False),
    "tenure": Column("int64", Check.in_range(0, 120), coerce=True),
    "PhoneService": Column(str, nullable=False),
    "MultipleLines": Column(str, nullable=False),
    "InternetService": Column(str, nullable=False),
    "OnlineSecurity": Column(str, nullable=False),
    "OnlineBackup": Column(str, nullable=False),
    "DeviceProtection": Column(str, nullable=False),
    "TechSupport": Column(str, nullable=False),
    "StreamingTV": Column(str, nullable=False),
    "StreamingMovies": Column(str, nullable=False),
    "Contract": Column(str, nullable=False),
    "PaperlessBilling": Column(str, nullable=False),
    "PaymentMethod": Column(str, nullable=False),
    "MonthlyCharges": Column("float64", Check.in_range(0, 1000), coerce=True),
    "TotalCharges": Column("float64", Check.in_range(0, 200_000), coerce=True, nullable=True),
}

INFERENCE_SCHEMA = DataFrameSchema(_INFERENCE_COLUMNS, coerce=False, strict=False)

#: The 19 raw model-input columns, in canonical order.
FEATURE_COLUMNS: list[str] = [c for c in _RAW_COLUMNS if c not in ("customerID", "Churn")]


class SchemaValidationError(ValueError):
    """Raised when a DataFrame fails schema validation."""

    def __init__(self, message: str, failure_cases: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.failure_cases = failure_cases


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a fully ingested dataset (identifier + target present)."""
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise SchemaValidationError(
            f"Raw dataset failed schema validation ({len(exc.failure_cases)} violations)",
            failure_cases=exc.failure_cases,
        ) from exc


def validate_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Validate inference-time feature frames (no identifier / target)."""
    try:
        return INFERENCE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise SchemaValidationError(
            f"Input failed schema validation ({len(exc.failure_cases)} violations)",
            failure_cases=exc.failure_cases,
        ) from exc
