"""Committed fixture corpus loading, keyed by image id."""

from __future__ import annotations

from typing import TypeVar

from rmo import paths
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

__all__ = ["load_records"]

_Record = TypeVar("_Record", OutfitDescription, OutfitScore, RoastOutput)


def load_records(name: str, model: type[_Record]) -> dict[str, _Record]:
    """Return the records of one corpus file, keyed by image id in file order."""
    lines = (paths.fixtures_dir() / name).read_text(encoding="utf-8").splitlines()
    records = [model.model_validate_json(line) for line in lines]
    return {record.image_id: record for record in records}
