"""Application configuration management.

Loads a YAML config file and applies environment-variable overrides.
Secrets are *never* read from config files; they come exclusively from the
environment (see `.env.example`).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def resolve_path(path: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_dir: Path
    source_url: str
    sha256: str
    target: str
    id_column: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    validation_size: float
    stratify: bool


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int
    cv_folds: int
    n_trials_lgbm: int
    n_trials_hgb: int
    n_trials_rf: int
    n_trials_lr: int


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    threshold_metric: str
    min_improvement_over_baseline: float


@dataclass(frozen=True)
class ModelConfig:
    candidates: list[str]
    models_dir: Path


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    rate_limit_per_minute: int
    max_batch_size: int
    max_body_bytes: int
    cors_origins: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MonitoringConfig:
    prediction_log_path: Path
    psi_warning: float
    psi_alert: float
    drift_min_samples: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    model: ModelConfig
    api: ApiConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML and apply environment overrides.

    Args:
        path: Optional explicit config path. Falls back to the ``CONFIG_PATH``
            environment variable, then ``configs/config.yaml``.

    Returns:
        Fully resolved, immutable application configuration.
    """
    config_path = resolve_path(path or _env("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    data = raw.get("data", {})
    split = raw.get("split", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    model = raw.get("model", {})
    api = raw.get("api", {})
    monitoring = raw.get("monitoring", {})
    logging_cfg = raw.get("logging", {})
    project = raw.get("project", {})

    cfg = AppConfig(
        project=ProjectConfig(
            name=project.get("name", "ml-system"),
            version=project.get("version", "0.0.0"),
        ),
        data=DataConfig(
            raw_path=resolve_path(_env("RAW_DATA_PATH") or data["raw_path"]),
            processed_dir=resolve_path(_env("PROCESSED_DIR") or data["processed_dir"]),
            source_url=data.get("source_url", ""),
            sha256=data.get("sha256", ""),
            target=data.get("target", "Churn"),
            id_column=data.get("id_column", "customerID"),
        ),
        split=SplitConfig(
            test_size=float(split.get("test_size", 0.15)),
            validation_size=float(split.get("validation_size", 0.1765)),
            stratify=bool(split.get("stratify", True)),
        ),
        training=TrainingConfig(
            random_state=int(training.get("random_state", 42)),
            cv_folds=int(training.get("cv_folds", 5)),
            n_trials_lgbm=int(training.get("n_trials_lgbm", 30)),
            n_trials_hgb=int(training.get("n_trials_hgb", 25)),
            n_trials_rf=int(training.get("n_trials_rf", 20)),
            n_trials_lr=int(training.get("n_trials_lr", 12)),
        ),
        evaluation=EvaluationConfig(
            primary_metric=evaluation.get("primary_metric", "f1"),
            threshold_metric=evaluation.get("threshold_metric", "f1"),
            min_improvement_over_baseline=float(
                evaluation.get("min_improvement_over_baseline", 0.05)
            ),
        ),
        model=ModelConfig(
            candidates=list(model.get("candidates", ["logreg"])),
            models_dir=resolve_path(_env("MODEL_DIR") or model.get("models_dir", "models")),
        ),
        api=ApiConfig(
            host=_env("API_HOST") or api.get("host", "0.0.0.0"),
            port=int(_env("API_PORT") or api.get("port", 8000)),
            rate_limit_per_minute=int(api.get("rate_limit_per_minute", 120)),
            max_batch_size=int(api.get("max_batch_size", 64)),
            max_body_bytes=int(api.get("max_body_bytes", 256 * 1024)),
            cors_origins=list(api.get("cors_origins", [])),
        ),
        monitoring=MonitoringConfig(
            prediction_log_path=resolve_path(
                _env("PREDICTION_LOG_PATH")
                or monitoring.get("prediction_log_path", "data/prediction_logs/predictions.jsonl")
            ),
            psi_warning=float(monitoring.get("psi_warning", 0.10)),
            psi_alert=float(monitoring.get("psi_alert", 0.20)),
            drift_min_samples=int(monitoring.get("drift_min_samples", 50)),
        ),
        logging=LoggingConfig(level=_env("LOG_LEVEL") or logging_cfg.get("level", "INFO")),
    )
    return cfg
