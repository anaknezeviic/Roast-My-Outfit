"""Filesystem layout resolution.

``$RMO_ROOT`` and ``$RMO_DATA_ROOT`` override the defaults. Asking for a path
never creates it; call :func:`ensure_dir` before writing.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_root",
    "ensure_dir",
    "raw_dir",
]

_ROOT_SENTINEL = "pyproject.toml"


def repo_root() -> Path:
    """Return ``$RMO_ROOT``, else the nearest ancestor containing ``pyproject.toml``."""
    override = os.environ.get("RMO_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise NotADirectoryError(
                f"RMO_ROOT is set to {candidate}, which is not a directory."
            )
        return candidate

    for candidate in Path(__file__).resolve().parents:
        if (candidate / _ROOT_SENTINEL).is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate {_ROOT_SENTINEL} in any parent of {__file__}. "
        "If rmo is installed non-editable, set the RMO_ROOT environment variable."
    )


def data_root() -> Path:
    """Return ``$RMO_DATA_ROOT`` if set, else ``repo_root() / "data"``."""
    override = os.environ.get("RMO_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and any missing parents, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    """Return ``<data_root>/raw`` — the dataset exactly as distributed."""
    return data_root() / "raw"
