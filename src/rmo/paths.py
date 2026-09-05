"""Filesystem layout and repository identity.

``$RMO_ROOT`` and ``$RMO_DATA_ROOT`` override the defaults. Asking for a path
never creates it; call :func:`ensure_dir` before writing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "repo_root",
    "data_root",
    "ensure_dir",
    "export_path",
    "file_sha256",
    "write_bytes_atomic",
    "write_json_atomic",
    "git_sha",
    "configs_dir",
    "raw_dir",
    "parsing_dir",
    "processed_dir",
    "splits_dir",
    "predictions_dir",
    "fixtures_dir",
    "results_dir",
    "metrics_dir",
]

_ROOT_SENTINEL = "pyproject.toml"
_CHUNK = 1 << 20


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


def export_path(path: str | Path | None) -> str:
    """Return a forward-slash repo-relative path, empty when it is under neither root."""
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        pass
    try:
        return f"data/{resolved.relative_to(data_root()).as_posix()}"
    except ValueError:
        return ""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> Path:
    """Replace ``path`` through a temporary sibling so a failed write leaves no partial file."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as sorted, finite JSON with a trailing newline."""
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ensure_dir(path.parent)
    return write_bytes_atomic(path, text.encode("utf-8"))


def git_sha() -> str | None:
    """Return the current commit, or None outside a work tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def configs_dir() -> Path:
    """Return ``<repo_root>/configs`` — prompts and lookup tables."""
    return repo_root() / "configs"


def raw_dir() -> Path:
    """Return ``<data_root>/raw`` — the dataset exactly as distributed."""
    return data_root() / "raw"


def parsing_dir() -> Path:
    """Return ``<raw_dir>/parsing`` — the per-image garment label maps."""
    return raw_dir() / "parsing"


def processed_dir() -> Path:
    """Return ``<data_root>/processed`` — tables derived from the raw dataset."""
    return data_root() / "processed"


def splits_dir() -> Path:
    """Return ``<data_root>/processed/splits`` — the frozen group-aware splits."""
    return processed_dir() / "splits"


def predictions_dir() -> Path:
    """Return ``<data_root>/processed/predictions`` — exported model outputs."""
    return processed_dir() / "predictions"


def fixtures_dir() -> Path:
    """Return ``<data_root>/fixtures`` — the committed fixture corpus."""
    return data_root() / "fixtures"


def results_dir() -> Path:
    """Return ``<repo_root>/results`` — committed report artefacts."""
    return repo_root() / "results"


def metrics_dir() -> Path:
    """Return ``<results_dir>/metrics`` — one JSON record per measured run."""
    return results_dir() / "metrics"
