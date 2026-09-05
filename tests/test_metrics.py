"""Check the shared run stamp for evaluation artefacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rmo.eval import metrics
from rmo.eval.metrics import (
    config_hash,
    metric_record,
    undefined,
    write_metric_record,
)
from rmo.schemas import SCHEMA_VERSION, Provenance

CONFIG = {"model_id": "smol", "prompt": "décris la tenue", "temperature": 0.0}

ENVELOPE_KEYS = {
    "baseline",
    "config_hash",
    "git_sha",
    "inputs",
    "metrics",
    "model",
    "n_items",
    "provenance",
    "schema_version",
    "seed",
    "split",
    "stage",
    "timestamp",
}


def record(**overrides: Any) -> dict[str, Any]:
    """Build a valid metric record, overriding any keyword."""
    arguments: dict[str, Any] = {
        "stage": "perception",
        "model": "smol",
        "split": "test",
        "n_items": 3,
        "metrics": {"accuracy": 0.5},
        "config": CONFIG,
    }
    arguments.update(overrides)
    return metric_record(**arguments)


@pytest.mark.parametrize("config", [CONFIG, {}])
def test_config_hash_matches_the_documented_digest(config: dict[str, Any]) -> None:
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    assert config_hash(config) == expected


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_separates_different_values() -> None:
    assert config_hash({**CONFIG, "temperature": 0.7}) != config_hash(CONFIG)


def test_config_hash_rejects_an_unserializable_config() -> None:
    with pytest.raises(TypeError):
        config_hash({"p": Path(".")})


def test_undefined_carries_null_and_reason() -> None:
    assert undefined("no_supervised_rows") == {
        "reason": "no_supervised_rows",
        "value": None,
    }


@pytest.mark.parametrize("reason", ["", "  "])
def test_undefined_rejects_a_blank_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        undefined(reason)


def test_metric_record_carries_the_full_envelope() -> None:
    stamp = record()
    assert set(stamp) == ENVELOPE_KEYS
    assert stamp["schema_version"] == SCHEMA_VERSION
    assert stamp["baseline"] is None
    assert stamp["inputs"] is None
    assert stamp["git_sha"] is None or re.fullmatch(r"[0-9a-f]{40}", stamp["git_sha"])
    digests = {"gold": "f" * 64, "split": "a" * 64}
    assert record(inputs=digests)["inputs"] == digests


def test_metric_record_uses_the_module_level_config_hash(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "config_hash", lambda config: "deadbeef")
    assert record()["config_hash"] == "deadbeef"


def test_metric_record_formats_the_timestamp_as_utc() -> None:
    stamped = datetime(2026, 9, 5, 14, 30, 15, tzinfo=UTC)
    assert record(timestamp=stamped)["timestamp"] == "2026-09-05T14:30:15Z"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record()["timestamp"])


def test_metric_record_defaults_provenance_to_the_enum_value() -> None:
    stamp = record()
    assert type(stamp["provenance"]) is str
    assert f"{stamp['provenance']}" == "predicted"
    assert stamp["provenance"] == Provenance.predicted.value


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"metrics": {}}, "at least one measured metric"),
        ({"n_items": 0}, "at least one measured item"),
        ({"stage": "captioning"}, "perception, scoring, roast"),
        ({"model": "  "}, "model identity"),
        ({"split": " "}, "split name"),
        ({"provenance": ""}, "needs a provenance"),
        ({"timestamp": datetime(2026, 9, 5, 14, 30, 15)}, "timezone-aware"),
    ],
)
def test_metric_record_rejects_an_unusable_stamp(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        record(**overrides)


def test_write_metric_record_is_sorted_lf_and_newline_terminated(tmp_path: Path) -> None:
    destination = write_metric_record(
        {"zebra": 1, "alpha": 2, "middle": 3}, tmp_path / "perception.json"
    )
    payload = destination.read_bytes()
    assert b"\r" not in payload
    assert payload.endswith(b"\n")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert list(json.loads(payload.decode("utf-8"))) == ["alpha", "middle", "zebra"]


def test_write_metric_record_rejects_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "nonfinite.json"
    with pytest.raises(ValueError):
        write_metric_record(record(metrics={"accuracy": float("nan")}), destination)
    assert not destination.parent.exists()


def test_write_metric_record_creates_its_directory_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "results" / "metrics" / "perception.json"
    write_metric_record(record(metrics={"accuracy": 0.25}), destination)
    write_metric_record(record(metrics={"accuracy": 0.75}), destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["metrics"] == {"accuracy": 0.75}
    assert list(destination.parent.iterdir()) == [destination]
