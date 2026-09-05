"""Export and re-read model predictions with a verifiable sidecar manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rmo import paths
from rmo.eval.metrics import config_hash, file_sha256, split_inputs
from rmo.imaging import IN_MEMORY_ID
from rmo.schemas import SCHEMA_VERSION, OutfitDescription
from rmo.splits import SPLIT_NAMES

__all__ = [
    "MANIFEST_VERSION",
    "PredictionError",
    "check_prediction_path",
    "read_predictions",
    "write_predictions",
]

MANIFEST_VERSION = 1

_MANIFEST_SUFFIX = ".manifest.json"


class PredictionError(ValueError):
    """Raised when a prediction file and its manifest disagree."""


def _manifest_path(destination: Path) -> Path:
    """Return the sidecar that sits beside one prediction file."""
    return destination.with_suffix(_MANIFEST_SUFFIX)


def check_prediction_path(destination: Path) -> None:
    """Raise when a prediction path would collide with a sidecar manifest."""
    if destination.suffix != ".jsonl":
        raise ValueError(
            f"A prediction file must end in .jsonl, got {destination.name!r}; "
            "any other suffix collides with an existing sidecar manifest."
        )


def write_predictions(
    descriptions: Sequence[OutfitDescription],
    destination: Path,
    *,
    split: str,
    model: str,
    config: Mapping[str, Any],
    expected_ids: set[str] | None = None,
    seed: int | None = None,
) -> Path:
    """Write predictions as JSONL plus a sidecar manifest and return the JSONL path."""
    if not descriptions:
        raise ValueError("Refusing to export a prediction file with no records.")
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {', '.join(SPLIT_NAMES)}.")
    check_prediction_path(destination)

    ordered = sorted(descriptions, key=lambda record: record.image_id)
    image_ids: set[str] = set()
    for record in ordered:
        if record.image_id == IN_MEMORY_ID:
            raise ValueError(
                f"Refusing to export a prediction keyed {IN_MEMORY_ID!r}: an image supplied "
                "as an array or an unnamed PIL image carries no stable id, so two of them "
                "collide in one file. Export from a file path or a split id instead."
            )
        if record.image_id in image_ids:
            raise ValueError(f"Duplicate prediction for {record.image_id!r}.")
        image_ids.add(record.image_id)

    if expected_ids is not None and image_ids != expected_ids:
        missing = sorted(expected_ids - image_ids)
        extra = sorted(image_ids - expected_ids)
        raise ValueError(
            f"Predictions do not cover {split!r}: {len(missing)} missing "
            f"({', '.join(missing) or 'none'}), {len(extra)} unexpected "
            f"({', '.join(extra) or 'none'})."
        )

    payload = "".join(f"{record.model_dump_json()}\n" for record in ordered).encode("utf-8")
    paths.ensure_dir(destination.parent)
    existed = destination.exists()
    paths.write_bytes_atomic(destination, payload)

    manifest = {
        "config_hash": config_hash(config),
        "git_sha": paths.git_sha(),
        "inputs": split_inputs(split),
        "manifest_version": MANIFEST_VERSION,
        "model": model,
        "n_records": len(ordered),
        "predictions_file": destination.name,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "split": split,
    }
    try:
        paths.write_json_atomic(_manifest_path(destination), manifest)
    except Exception:
        if not existed:
            destination.unlink(missing_ok=True)
        raise
    return destination


def read_predictions(source: Path) -> list[OutfitDescription]:
    """Return validated records after checking them against the sidecar manifest."""
    manifest_path = _manifest_path(source)
    if not manifest_path.is_file():
        raise PredictionError(f"{source} has no manifest at {manifest_path.name}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PredictionError(f"The manifest for {source} is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise PredictionError(f"The manifest for {source} is not a JSON object.")

    version = manifest.get("manifest_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or not 1 <= version <= MANIFEST_VERSION
    ):
        raise PredictionError(
            f"The manifest for {source} declares version {version!r}, "
            f"and this reader understands {MANIFEST_VERSION}."
        )
    if manifest.get("predictions_file") != source.name:
        raise PredictionError(
            f"The manifest for {source} belongs to "
            f"{manifest.get('predictions_file')!r} instead."
        )
    if not source.is_file():
        raise PredictionError(f"{source} is named by its manifest but does not exist.")
    if file_sha256(source) != manifest.get("sha256"):
        raise PredictionError(f"{source} does not match the sha256 recorded in its manifest.")

    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PredictionError(f"{source} is not valid UTF-8.") from exc
    # pydantic emits a literal U+2028, which str.splitlines() would split on.
    lines = text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")
    if len(lines) != manifest.get("n_records"):
        raise PredictionError(
            f"{source} holds {len(lines)} records and its manifest claims "
            f"{manifest.get('n_records')}."
        )

    records: list[OutfitDescription] = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(OutfitDescription.model_validate_json(line))
        except ValidationError as exc:
            raise PredictionError(f"{source} line {number} is not a valid description.") from exc

    image_ids = [record.image_id for record in records]
    if any(later <= earlier for earlier, later in zip(image_ids, image_ids[1:])):
        raise PredictionError(f"{source} image ids are not unique and strictly ascending.")
    return records
