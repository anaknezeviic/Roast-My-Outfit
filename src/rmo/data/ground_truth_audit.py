"""Audit staged labels against parsing-mask garment presence before VLM training.

This module is deliberately diagnostic: it does not choose how one-piece garments
(dresses and rompers) should inherit the dataset's upper/lower regional labels.
Instead it measures the evidence needed to make that mapping explicit and safe.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from rmo.config import load_perception_config
from rmo.data.descriptions import load_outfit_table
from rmo.data.label_resolution import ONE_PIECE_SLOTS, resolve_regional_value
from rmo.perception.enrichment import mask_path, mask_skeleton, slot_labels
from rmo.schemas import GarmentSlot

__all__ = [
    "GroundTruthAuditError",
    "PairAudit",
    "OnePieceAudit",
    "GroundTruthAudit",
    "audit_ground_truth",
]

_NA = "na"
_REGIONAL_TEXTURE_COLUMNS: dict[GarmentSlot, tuple[str, str]] = {
    GarmentSlot.upper: ("upper_fabric", "upper_pattern"),
    GarmentSlot.outer: ("outer_fabric", "outer_pattern"),
    GarmentSlot.lower: ("lower_fabric", "lower_pattern"),
}
_REQUIRED_COLUMNS = {
    "upper_fabric",
    "lower_fabric",
    "outer_fabric",
    "upper_pattern",
    "lower_pattern",
    "outer_pattern",
    "has_shape",
}


class GroundTruthAuditError(RuntimeError):
    """Raised when the staged corpus cannot be audited reliably."""


@dataclass(frozen=True, slots=True)
class PairAudit:
    """Relationship between the dataset's upper and lower regional labels."""

    total: int
    both_na: int
    upper_only: int
    lower_only: int
    same: int
    conflict: int

    @classmethod
    def from_counter(cls, counter: Mapping[str, int]) -> PairAudit:
        return cls(
            total=sum(counter.values()),
            both_na=counter.get("both_na", 0),
            upper_only=counter.get("upper_only", 0),
            lower_only=counter.get("lower_only", 0),
            same=counter.get("same", 0),
            conflict=counter.get("conflict", 0),
        )


@dataclass(frozen=True, slots=True)
class OnePieceAudit:
    """Evidence available for one one-piece garment slot."""

    images: int
    shape_annotated: int
    fabric: PairAudit
    pattern: PairAudit


@dataclass(frozen=True, slots=True)
class GroundTruthAudit:
    """Summary of garment presence and regional-label consistency."""

    images: int
    slot_presence: dict[str, int]
    regional_texture_both_na: dict[str, int]
    dress: OnePieceAudit
    romper: OnePieceAudit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _read_mask(image_id: str) -> np.ndarray:
    path = mask_path(image_id)
    if path is None:
        raise GroundTruthAuditError(f"No parsing mask staged for {image_id!r}.")
    try:
        with Image.open(path) as handle:
            array = np.asarray(handle)
    except (OSError, ValueError) as exc:
        raise GroundTruthAuditError(f"Could not read parsing mask {path}: {exc}.") from exc
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise GroundTruthAuditError(
            f"Parsing mask {path} has {array.ndim} dimensions; expected a 2-D label map."
        )
    return array


def audit_ground_truth(
    image_ids: Iterable[str] | None = None,
    *,
    table: pd.DataFrame | None = None,
    config_path: Path | None = None,
) -> GroundTruthAudit:
    """Audit parsing-mask presence against staged texture/shape annotations.

    ``image_ids`` defaults to the entire canonical table. Every requested id must
    exist in the table and have a readable parsing mask; the audit fails rather
    than silently dropping data.
    """
    frame = load_outfit_table() if table is None else table
    if "image_id" in frame.columns and frame.index.name != "image_id":
        frame = frame.set_index("image_id", drop=False)

    missing_columns = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise GroundTruthAuditError(
            f"Outfit table is missing required audit columns: {', '.join(missing_columns)}."
        )

    ids = list(frame.index if image_ids is None else image_ids)
    duplicates = [name for name, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise GroundTruthAuditError(
            f"Audit input repeats {len(duplicates)} image ids; first: {duplicates[0]!r}."
        )

    missing = [name for name in ids if name not in frame.index]
    if missing:
        raise GroundTruthAuditError(
            f"{len(missing)} audit ids are absent from the outfit table; first: {missing[0]!r}."
        )

    labels = slot_labels(load_perception_config(config_path))
    slot_counts: Counter[GarmentSlot] = Counter()
    regional_na_na: Counter[GarmentSlot] = Counter()
    one_piece_images: Counter[GarmentSlot] = Counter()
    one_piece_shape: Counter[GarmentSlot] = Counter()
    one_piece_fabric: dict[GarmentSlot, Counter[str]] = {
        slot: Counter() for slot in ONE_PIECE_SLOTS
    }
    one_piece_pattern: dict[GarmentSlot, Counter[str]] = {
        slot: Counter() for slot in ONE_PIECE_SLOTS
    }

    for image_id in ids:
        row = frame.loc[image_id]
        if isinstance(row, pd.DataFrame):
            raise GroundTruthAuditError(f"Outfit table repeats image id {image_id!r}.")

        mask = _read_mask(image_id)
        present = {slot for slot, _ in mask_skeleton(mask, labels)}
        if not present:
            raise GroundTruthAuditError(
                f"Parsing mask for {image_id!r} contains no configured garment slot."
            )
        slot_counts.update(present)

        for slot, (fabric_column, pattern_column) in _REGIONAL_TEXTURE_COLUMNS.items():
            if slot not in present:
                continue
            if str(row[fabric_column]) == _NA and str(row[pattern_column]) == _NA:
                regional_na_na[slot] += 1

        for slot in ONE_PIECE_SLOTS:
            if slot not in present:
                continue
            one_piece_images[slot] += 1
            if bool(row["has_shape"]):
                one_piece_shape[slot] += 1
            fabric = resolve_regional_value(row["upper_fabric"], row["lower_fabric"])
            one_piece_fabric[slot][fabric.kind] += 1
            pattern = resolve_regional_value(row["upper_pattern"], row["lower_pattern"])
            one_piece_pattern[slot][pattern.kind] += 1

    def one_piece(slot: GarmentSlot) -> OnePieceAudit:
        return OnePieceAudit(
            images=one_piece_images[slot],
            shape_annotated=one_piece_shape[slot],
            fabric=PairAudit.from_counter(one_piece_fabric[slot]),
            pattern=PairAudit.from_counter(one_piece_pattern[slot]),
        )

    return GroundTruthAudit(
        images=len(ids),
        slot_presence={slot.value: slot_counts[slot] for slot in GarmentSlot},
        regional_texture_both_na={
            slot.value: regional_na_na[slot] for slot in _REGIONAL_TEXTURE_COLUMNS
        },
        dress=one_piece(GarmentSlot.dress),
        romper=one_piece(GarmentSlot.romper),
    )
