"""Dataset ingestion.

Downloads the Telco Customer Churn CSV from the configured URL, verifies its
SHA-256 checksum against the pinned value in ``configs/config.yaml`` (supply-
chain protection against silent upstream changes), and stores it under
``data/raw/``.

Run:
    python -m src.data.ingestion [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import AppConfig, load_config
from src.data.schemas import validate_raw
from src.logging_utils import get_logger, setup_logging

logger = get_logger("ingestion")

_CHUNK_SIZE = 1 << 16
_DOWNLOAD_TIMEOUT_S = 120


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(cfg: AppConfig, force: bool = False) -> Path:
    """Download the raw CSV if missing (or forced) and verify its checksum."""
    raw_path = cfg.data.raw_path
    if raw_path.exists() and not force:
        logger.info("Raw dataset already present", extra={"path": str(raw_path)})
    else:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset", extra={"url": cfg.data.source_url})
        tmp_path = raw_path.with_suffix(".part")
        request = urllib.request.Request(
            cfg.data.source_url, headers={"User-Agent": "telco-churn-ml-system/1.0"}
        )
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            with open(tmp_path, "wb") as fh:
                while chunk := response.read(_CHUNK_SIZE):
                    fh.write(chunk)
        tmp_path.replace(raw_path)

    if cfg.data.sha256:
        actual = sha256_of(raw_path)
        if actual != cfg.data.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {raw_path}: expected {cfg.data.sha256}, got {actual}. "
                "Upstream data changed or the download is corrupt — investigate before proceeding."
            )
        logger.info("Checksum verified", extra={"sha256": actual})
    return raw_path


def load_raw_dataset(cfg: AppConfig) -> pd.DataFrame:
    """Load and schema-validate the raw dataset.

    TotalCharges is deliberately read as string because the upstream file uses
    blank cells (not NaN markers) for the 11 customers with tenure=0.
    """
    raw_path = cfg.data.raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run `python -m src.data.ingestion` first."
        )
    df = pd.read_csv(raw_path, dtype={"TotalCharges": "string"})
    df = validate_raw(df)
    logger.info(
        "Raw dataset loaded and validated",
        extra={"rows": len(df), "columns": len(df.columns)},
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify the raw dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.logging.level)
    path = download_dataset(cfg, force=args.force)
    df = load_raw_dataset(cfg)
    logger.info("Ingestion complete", extra={"path": str(path), "rows": len(df)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
