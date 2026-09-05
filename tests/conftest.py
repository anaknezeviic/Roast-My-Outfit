"""Shared access to the committed fixture corpus."""

from __future__ import annotations

import shutil
from typing import Any

import pytest
import yaml

from rmo import fixtures, paths
from rmo.config import load_scoring_config
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

EXPECTED_IDS = [
    *(f"fixture_{index:03d}" for index in range(17)),
    *(f"fx_adv_{index:02d}" for index in range(3)),
    *(f"fx_deg_{index:02d}" for index in range(8)),
]

SAMPLE_ID = "fixture_000"


@pytest.fixture(scope="module")
def descriptions() -> dict[str, OutfitDescription]:
    """Return the committed descriptions."""
    return fixtures.load_records("outfit_descriptions.jsonl", OutfitDescription)


@pytest.fixture(scope="module")
def scores() -> dict[str, OutfitScore]:
    """Return the committed scores."""
    return fixtures.load_records("outfit_scores.jsonl", OutfitScore)


@pytest.fixture(scope="module")
def roasts() -> dict[str, RoastOutput]:
    """Return the committed roasts."""
    return fixtures.load_records("roast_outputs.jsonl", RoastOutput)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` applied, recursing into nested mappings."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@pytest.fixture()
def scoring_config(tmp_path, monkeypatch):
    """Write a scoring config into a temporary repo root and clear the loader cache."""
    real_data_root = paths.data_root()
    shipped_configs = paths.configs_dir()
    shipped = yaml.safe_load((shipped_configs / "scoring.yaml").read_text(encoding="utf-8"))

    root = tmp_path / "repo"
    configs = root / "configs"
    shutil.copytree(shipped_configs, configs)

    def write(**overrides: Any) -> dict[str, Any]:
        payload = _deep_merge(shipped, overrides)
        (configs / "scoring.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
        monkeypatch.setenv("RMO_DATA_ROOT", str(real_data_root))
        monkeypatch.setenv("RMO_ROOT", str(root))
        load_scoring_config.cache_clear()
        return payload

    yield write
    load_scoring_config.cache_clear()
