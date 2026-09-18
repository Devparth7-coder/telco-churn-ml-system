"""API request/response schemas (the public contract).

Field names intentionally match the raw dataset columns so the contract is
unambiguous. Strict literal types + numeric bounds reject malformed payloads
at the edge with actionable 422 errors, before anything reaches the model.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """The 19 raw model inputs (no identifier, no target)."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0, le=200_000)

    def to_raw_dict(self) -> dict:
        """Convert to the raw ingestion representation (TotalCharges None -> NaN)."""
        data = self.model_dump()
        if data["TotalCharges"] is None:
            data["TotalCharges"] = pd.NA
        return data


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(default=None, max_length=64)
    features: CustomerFeatures


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)
    applied_threshold: float = Field(ge=0, le=1)
    model_version: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionRequest] = Field(min_length=1)


class BatchItemResult(BaseModel):
    index: int
    customer_id: str | None = None
    ok: bool
    result: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    results: list[BatchItemResult]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_version: str | None = None


class VersionResponse(BaseModel):
    service: str
    service_version: str
    model_version: str | None = None
    estimator: str | None = None


def features_to_dataframe(requests: list[PredictionRequest]) -> pd.DataFrame:
    """Stack request payloads into the raw feature frame the predictor expects."""
    return pd.DataFrame([request.features.to_raw_dict() for request in requests])
