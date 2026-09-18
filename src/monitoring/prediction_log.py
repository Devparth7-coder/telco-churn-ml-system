"""Append-only JSONL prediction log used for drift monitoring and auditing.

Records contain raw features, scores, and metadata — never request bodies,
headers, or credentials. Customer identifiers are truncated-hashed before
logging so the log stays pseudonymous.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def hash_customer_id(customer_id: str | None) -> str | None:
    """Pseudonymize an identifier (not reversible to the original)."""
    if customer_id in (None, ""):
        return None
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


class PredictionLogger:
    """Thread-safe JSONL writer/reader for prediction records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        line = json.dumps(record, default=str)
        with _LOCK, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_recent(self, n: int) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        tail: Iterable[str]
        with open(self.path, encoding="utf-8") as fh:
            tail = fh.readlines()[-max(n, 0):]
        records: list[dict[str, Any]] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn write; never crash monitoring
        return records
