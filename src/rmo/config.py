"""YAML configuration loading for models whose prompts and tables live in ``configs/``.

The parsed mapping is cached per path and shared; treat it as read-only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rmo import paths

__all__ = [
    "ConfigError",
    "load_perception_config",
    "load_roast_config",
    "load_scoring_config",
]


class ConfigError(ValueError):
    """Raised when a configuration file is missing or is not a YAML mapping."""


def _load_mapping(resolved: Path) -> dict[str, Any]:
    """Return the YAML mapping stored at ``resolved``."""
    try:
        with resolved.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read configuration from {resolved}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration file {resolved} must contain a YAML mapping.")
    return payload


@lru_cache(maxsize=None)
def load_perception_config(path: Path | None = None) -> dict[str, Any]:
    """Return the perception configuration, defaulting to ``configs/perception.yaml``."""
    return _load_mapping(paths.configs_dir() / "perception.yaml" if path is None else Path(path))


@lru_cache(maxsize=None)
def load_roast_config(path: Path | None = None) -> dict[str, Any]:
    """Return the roast configuration, defaulting to ``configs/roast.yaml``."""
    return _load_mapping(paths.configs_dir() / "roast.yaml" if path is None else Path(path))


@lru_cache(maxsize=None)
def load_scoring_config(path: Path | None = None) -> dict[str, Any]:
    """Return the scoring configuration, defaulting to ``configs/scoring.yaml``."""
    return _load_mapping(paths.configs_dir() / "scoring.yaml" if path is None else Path(path))
