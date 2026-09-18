"""API endpoint tests (validation, auth, limits, error handling)."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import load_config
from tests.conftest import make_payload, requires_dataset

pytestmark = requires_dataset


@pytest.fixture()
def client(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(trained_artifact.path))
    with TestClient(create_app(load_config())) as test_client:
        yield test_client


def _override_api_cfg(**api_kwargs):
    cfg = load_config()
    return dataclasses.replace(cfg, api=dataclasses.replace(cfg.api, **api_kwargs))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_loaded_model(client, trained_artifact):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_ready_not_ready_without_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(create_app(load_config())) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert test_client.post("/predict", json=make_payload()).status_code == 503


def test_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "telco-churn-ml-system"
    assert body["model_version"] == "test-fixture"


def test_predict_happy_path(client):
    response = client.post("/predict", json=make_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] == "test-fixture"
    assert body["request_id"]
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_predict_custom_request_id_echoed(client):
    response = client.post("/predict", json=make_payload(), headers={"X-Request-ID": "abc-123"})
    assert response.json()["request_id"] == "abc-123"


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenure": -1},
        {"MonthlyCharges": -5},
        {"gender": "Other"},
        {"Contract": "Lifetime"},
        {"SeniorCitizen": 2},
    ],
)
def test_predict_rejects_invalid_payloads(client, overrides):
    response = client.post("/predict", json=make_payload(**overrides))
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = make_payload()
    del payload["features"]["tenure"]
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_extra_fields(client):
    payload = make_payload()
    payload["features"]["ssn"] = "123-45-6789"
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_handles_missing_total_charges(client):
    payload = make_payload(TotalCharges=None)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_batch_happy_path(client):
    payload = {"items": [make_payload(), make_payload(tenure=48, Contract="Two year")]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert all(item["ok"] for item in body["results"])
    assert body["model_version"] == "test-fixture"


def test_batch_empty_rejected(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422


def test_batch_size_limit(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(trained_artifact.path))
    cfg = _override_api_cfg(max_batch_size=2)
    with TestClient(create_app(cfg)) as test_client:
        payload = {"items": [make_payload(), make_payload(), make_payload()]}
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 413


def test_rate_limit_enforced(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(trained_artifact.path))
    cfg = _override_api_cfg(rate_limit_per_minute=2)
    with TestClient(create_app(cfg)) as test_client:
        assert test_client.post("/predict", json=make_payload()).status_code == 200
        assert test_client.post("/predict", json=make_payload()).status_code == 200
        limited = test_client.post("/predict", json=make_payload())
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers


def test_auth_required_when_token_set(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(trained_artifact.path))
    monkeypatch.setenv("API_AUTH_TOKEN", "sekrit-token")
    with TestClient(create_app(load_config())) as test_client:
        assert test_client.post("/predict", json=make_payload()).status_code == 401
        assert (
            test_client.post("/predict", json=make_payload(), headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        ok = test_client.post(
            "/predict", json=make_payload(), headers={"Authorization": "Bearer sekrit-token"}
        )
        assert ok.status_code == 200
        # Health stays open for load balancers.
        assert test_client.get("/health").status_code == 200


def test_body_size_limit(trained_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(trained_artifact.path))
    cfg = _override_api_cfg(max_body_bytes=64)
    with TestClient(create_app(cfg)) as test_client:
        response = test_client.post("/predict", json=make_payload())
        assert response.status_code == 413
