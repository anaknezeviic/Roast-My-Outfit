"""Shared measurement stamping for evaluation artefacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rmo import paths
from rmo.paths import file_sha256
from rmo.schemas import SCHEMA_VERSION, Provenance

__all__ = [
    "STAGES",
    "config_hash",
    "file_sha256",
    "metric_record",
    "split_inputs",
    "undefined",
    "write_metric_record",
]

STAGES: tuple[str, ...] = ("perception", "scoring", "roast")

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the first eight hex characters of the sorted compact-JSON digest."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def undefined(reason: str) -> dict[str, Any]:
    """Return the null-plus-reason placeholder for a metric that cannot be measured."""
    if not reason.strip():
        raise ValueError("An undefined metric needs a stated reason.")
    return {"reason": reason, "value": None}


def split_inputs(split: str) -> dict[str, str]:
    """Return the digest of the committed split file, empty when it is not staged."""
    path = paths.splits_dir() / f"{split}.txt"
    return {"split": file_sha256(path)} if path.is_file() else {}


def metric_record(
    *,
    stage: str,
    model: str,
    split: str,
    n_items: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: str = Provenance.predicted.value,
    baseline: Mapping[str, Any] | None = None,
    seed: int | None = None,
    inputs: Mapping[str, str] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Return the shared run stamp around one set of measured metrics."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}.")
    if not model.strip():
        raise ValueError("A metric record needs a model identity.")
    if not split.strip():
        raise ValueError("A metric record needs a split name.")
    if not provenance.strip():
        raise ValueError("A metric record needs a provenance.")
    if n_items < 1:
        raise ValueError(f"A metric record needs at least one measured item, got {n_items}.")
    if not metrics:
        raise ValueError("A metric record needs at least one measured metric.")

    stamped = timestamp if timestamp is not None else datetime.now(UTC)
    if stamped.tzinfo is None or stamped.utcoffset() is None:
        raise ValueError("A metric record needs a timezone-aware timestamp.")

    return {
        "baseline": dict(baseline) if baseline is not None else None,
        "config_hash": config_hash(config),
        "git_sha": paths.git_sha(),
        "inputs": dict(inputs) if inputs is not None else None,
        "metrics": dict(metrics),
        "model": model,
        "n_items": n_items,
        "provenance": provenance,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split": split,
        "stage": stage,
        "timestamp": stamped.astimezone(UTC).strftime(_TIMESTAMP_FORMAT),
    }


def write_metric_record(record: Mapping[str, Any], destination: Path) -> Path:
    """Write one metric record as sorted JSON and return its path."""
    return paths.write_json_atomic(destination, record)
