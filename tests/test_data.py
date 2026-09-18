"""Data validation and splitting tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import sha256_of
from src.data.preprocessing import CleaningTransformer
from src.data.schemas import (
    FEATURE_COLUMNS,
    SchemaValidationError,
    validate_inference,
    validate_raw,
)
from src.data.splitting import stratified_split
from tests.conftest import requires_dataset


@requires_dataset
def test_raw_schema_accepts_valid_dataset(raw_data):
    validated = validate_raw(raw_data)
    assert len(validated) == len(raw_data)


def test_raw_schema_rejects_invalid_target(raw_data):
    bad = raw_data.copy()
    bad.loc[bad.index[0], "Churn"] = "Maybe"
    with pytest.raises(SchemaValidationError):
        validate_raw(bad)


def test_raw_schema_rejects_negative_charges(raw_data):
    bad = raw_data.copy()
    bad.loc[bad.index[0], "MonthlyCharges"] = -5.0
    with pytest.raises(SchemaValidationError):
        validate_raw(bad)


def test_inference_schema_rejects_missing_column(raw_data):
    frame = raw_data[FEATURE_COLUMNS].drop(columns=["Contract"])
    with pytest.raises(SchemaValidationError):
        validate_inference(frame)


def test_cleaning_transformer_coerces_total_charges():
    frame = pd.DataFrame(
        {
            "TotalCharges": ["100.5", "", None, "200"],
            "SeniorCitizen": [0, 1, 0, 1],
            "gender": [" Male", "Female", "Male", "Female"],
        }
    )
    out = CleaningTransformer().fit_transform(frame)
    assert out["TotalCharges"].tolist()[0] == pytest.approx(100.5)
    assert pd.isna(out["TotalCharges"].iloc[1])
    assert pd.isna(out["TotalCharges"].iloc[2])
    assert out["gender"].iloc[0] == "Male"  # whitespace stripped


@requires_dataset
def test_split_is_disjoint_complete_and_stratified(raw_data, cfg):
    split = stratified_split(raw_data, "Churn", "customerID", cfg.split, random_state=42)
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(raw_data)

    ids = lambda df: set(df["customerID"])  # noqa: E731
    assert not (ids(split.train) & ids(split.validation))
    assert not (ids(split.train) & ids(split.test))
    assert not (ids(split.validation) & ids(split.test))

    overall_rate = (raw_data["Churn"] == "Yes").mean()
    for part in (split.train, split.validation, split.test):
        assert abs((part["Churn"] == "Yes").mean() - overall_rate) < 0.02

    assert len(split.assignments) == total


@requires_dataset
def test_split_rejects_duplicate_identifiers(raw_data, cfg):
    duplicated = pd.concat([raw_data, raw_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate identifiers"):
        stratified_split(duplicated, "Churn", "customerID", cfg.split, random_state=42)


def test_checksum_detects_corruption(tmp_path: Path):
    file_path = tmp_path / "payload.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    expected = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert sha256_of(file_path) == expected
