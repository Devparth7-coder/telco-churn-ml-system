"""FastAPI service exposing churn predictions.

Endpoints: GET /health, GET /ready, GET /version, POST /predict,
POST /predict/batch. The model artifact is loaded exactly once during
application startup (lifespan) and never reloaded per request.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchItemResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    VersionResponse,
)
from src.config import AppConfig, load_config
from src.data.schemas import SchemaValidationError
from src.logging_utils import get_logger, setup_logging
from src.models.predict import Predictor
from src.monitoring.prediction_log import PredictionLogger, hash_customer_id

logger = get_logger("api")

#: Canonical record used for the startup self-check (also handy in docs/tests).
SMOKE_FEATURES: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class RateLimiter:
    """In-process fixed-window rate limiter keyed by client IP."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start, count = self._buckets.get(key, (now, 0))
        if now - window_start >= 60.0:
            window_start, count = now, 0
        count += 1
        self._buckets[key] = (window_start, count)
        if len(self._buckets) > 10_000:  # bounded memory under hostile traffic
            self._buckets = dict(sorted(self._buckets.items(), key=lambda kv: kv[1][0])[-5_000:])
        return count <= self.limit


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _authorized(request: Request) -> bool:
    expected = os.environ.get("API_AUTH_TOKEN", "")
    if not expected:
        return True  # auth disabled; documented in README
    supplied = request.headers.get("authorization", "")
    if not supplied.startswith("Bearer "):
        return False
    return secrets.compare_digest(supplied.removeprefix("Bearer ").strip(), expected)


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Application factory (tests build isolated instances with their own config)."""
    cfg = cfg or load_config()
    setup_logging(cfg.logging.level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        predictor: Predictor | None = None
        try:
            predictor = Predictor.load_default()
            smoke = predictor.predict_record(SMOKE_FEATURES)
            logger.info(
                "Model loaded and startup self-check passed",
                extra={
                    "model_version": predictor.model_version,
                    "threshold": predictor.threshold,
                    "smoke_probability": round(smoke["churn_probability"], 4),
                },
            )
        except Exception as exc:  # noqa: BLE001 — startup must degrade to /ready=false, not crash-loop info
            logger.error("Model failed to load; service will report not ready", extra={"error": str(exc)})
            predictor = None

        app.state.predictor = predictor
        app.state.config = cfg
        app.state.rate_limiter = RateLimiter(cfg.api.rate_limit_per_minute)
        app.state.prediction_logger = PredictionLogger(cfg.monitoring.prediction_log_path)
        yield

    app = FastAPI(
        title="Telco Churn Prediction API",
        version=cfg.project.version,
        lifespan=lifespan,
    )

    cors_origins = [o.strip() for o in os.environ.get("API_CORS_ORIGINS", "").split(",") if o.strip()] or cfg.api.cors_origins
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def operational_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        # Request size cap (defense against oversized/malicious payloads).
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > cfg.api.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
                headers={"X-Request-ID": request_id},
            )

        if request.url.path in ("/predict", "/predict/batch"):
            if not _authorized(request):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing bearer token"},
                    headers={"X-Request-ID": request_id},
                )
            if not app.state.rate_limiter.allow(_client_ip(request)):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": "60", "X-Request-ID": request_id},
                )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled error",
                extra={"request_id": request_id, "path": request.url.path},
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"}, headers={"X-Request-ID": request_id})

        latency_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response

    @app.exception_handler(SchemaValidationError)
    async def schema_validation_handler(request: Request, exc: SchemaValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "violations": exc.failure_cases.head(20).to_dict(orient="records")
                if exc.failure_cases is not None
                else [],
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    def _require_predictor(request: Request) -> Predictor:
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="Model not loaded — service not ready")
        return predictor

    def _score_request(
        predictor: Predictor, payload: PredictionRequest, request_id: str, request: Request
    ) -> PredictionResponse:
        started = time.perf_counter()
        result = predictor.predict_record(payload.features.to_raw_dict())
        latency_ms = (time.perf_counter() - started) * 1000

        request.app.state.prediction_logger.log(
            {
                "request_id": request_id,
                "customer_id_hash": hash_customer_id(payload.customer_id),
                "features": {k: (None if v is None else v) for k, v in payload.features.to_raw_dict().items()},
                "churn_probability": round(result["churn_probability"], 6),
                "prediction": result["prediction"],
                "model_version": predictor.model_version,
                "inference_ms": round(latency_ms, 2),
            }
        )
        return PredictionResponse(
            customer_id=payload.customer_id,
            prediction=result["prediction"],  # type: ignore[arg-type]
            churn_probability=round(result["churn_probability"], 6),
            applied_threshold=predictor.threshold,
            model_version=predictor.model_version,
            request_id=request_id,
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness: the process is up (does not depend on the model)."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    async def ready(request: Request) -> ReadinessResponse:
        """Readiness: a model artifact is loaded and serving."""
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            return JSONResponse(  # type: ignore[return-value]
                status_code=503,
                content=ReadinessResponse(status="not_ready", model_loaded=False, model_version=None).model_dump(),
            )
        return ReadinessResponse(status="ready", model_loaded=True, model_version=predictor.model_version)

    @app.get("/version", response_model=VersionResponse, tags=["ops"])
    async def version(request: Request) -> VersionResponse:
        predictor: Predictor | None = request.app.state.predictor
        return VersionResponse(
            service=cfg.project.name,
            service_version=cfg.project.version,
            model_version=predictor.model_version if predictor else None,
            estimator=predictor.artifact.meta.get("estimator") if predictor else None,
        )

    def _score_batch_vectorized(
        predictor: Predictor, payload: BatchPredictionRequest, request: Request
    ) -> list[BatchItemResult] | None:
        """Score the whole batch in one pipeline call.

        Returns ``None`` when the vectorized pass fails so the caller can fall
        back to per-item scoring and attribute errors to individual rows.
        """
        from src.api.schemas import features_to_dataframe

        started = time.perf_counter()
        try:
            frame = features_to_dataframe(payload.items)
            scored = predictor.predict_dataframe(frame)
        except (SchemaValidationError, ValueError):
            return None
        latency_ms = (time.perf_counter() - started) * 1000

        logger_handle = request.app.state.prediction_logger
        results: list[BatchItemResult] = []
        for index, (item, (_, row)) in enumerate(zip(payload.items, scored.iterrows(), strict=True)):
            probability = float(row["churn_probability"])
            logger_handle.log(
                {
                    "request_id": request.state.request_id,
                    "customer_id_hash": hash_customer_id(item.customer_id),
                    "features": {
                        k: (None if v is None else v) for k, v in item.features.to_raw_dict().items()
                    },
                    "churn_probability": round(probability, 6),
                    "prediction": str(row["prediction"]),
                    "model_version": predictor.model_version,
                    "inference_ms": round(latency_ms / len(payload.items), 3),
                }
            )
            results.append(
                BatchItemResult(
                    index=index,
                    customer_id=item.customer_id,
                    ok=True,
                    result=PredictionResponse(
                        customer_id=item.customer_id,
                        prediction=str(row["prediction"]),  # type: ignore[arg-type]
                        churn_probability=round(probability, 6),
                        applied_threshold=predictor.threshold,
                        model_version=predictor.model_version,
                        request_id=request.state.request_id,
                    ),
                )
            )
        return results

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        predictor = _require_predictor(request)
        return _score_request(predictor, payload, request.state.request_id, request)

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    async def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
        predictor = _require_predictor(request)
        if len(payload.items) > cfg.api.max_batch_size:
            return JSONResponse(  # type: ignore[return-value]
                status_code=413,
                content={"detail": f"Batch exceeds maximum size of {cfg.api.max_batch_size} items"},
            )

        results = _score_batch_vectorized(predictor, payload, request)
        if results is None:
            # Slow path: attribute failures to individual items.
            results = []
            for index, item in enumerate(payload.items):
                try:
                    response = _score_request(predictor, item, request.state.request_id, request)
                    results.append(
                        BatchItemResult(index=index, customer_id=item.customer_id, ok=True, result=response)
                    )
                except SchemaValidationError as exc:
                    results.append(BatchItemResult(index=index, customer_id=item.customer_id, ok=False, error=str(exc)))
                except ValueError as exc:
                    results.append(BatchItemResult(index=index, customer_id=item.customer_id, ok=False, error=str(exc)))

        return BatchPredictionResponse(
            results=results, model_version=predictor.model_version, request_id=request.state.request_id
        )

    return app


app = create_app()
