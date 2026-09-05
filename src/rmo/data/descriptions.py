"""Build ground-truth outfit descriptions from labels, parsing masks and pixels.

Masks decide which garments are present, the label columns supply their attributes
and the photograph supplies colour.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from rmo import paths
from rmo.config import load_perception_config
from rmo.data.preflight import photo_path, report_for_ids, require
from rmo.imaging import load_image
from rmo.perception.enrichment import (
    HEAD_PROJECTION,
    SLOT_CATEGORIES,
    apply_palette,
    load_mask,
    mask_skeleton,
    slot_labels,
)
from rmo.schemas import (
    Fabric,
    Garment,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
    Provenance,
    SleeveLength,
)
from rmo.splits import load_split

log = logging.getLogger(__name__)

__all__ = [
    "SOURCE_MODEL",
    "DescriptionError",
    "describe_image",
    "describe_split",
    "load_outfit_table",
]

SOURCE_MODEL = "dataset_labels"

_TEXTURE_COLUMNS: dict[GarmentSlot, tuple[str, str]] = {
    GarmentSlot.upper: ("upper_fabric", "upper_pattern"),
    GarmentSlot.outer: ("outer_fabric", "outer_pattern"),
    GarmentSlot.lower: ("lower_fabric", "lower_pattern"),
}

_SLEEVED = HEAD_PROJECTION["sleeve_length"][1]
_HEMMED = HEAD_PROJECTION["lower_length"][1]

_NA = "na"


class DescriptionError(ValueError):
    """Raised when a staged row cannot produce a valid description."""


def load_outfit_table(path: Path | None = None) -> pd.DataFrame:
    """Return the canonical outfit table indexed by image id."""
    resolved = path or paths.processed_dir() / "outfits.parquet"
    if not resolved.is_file():
        raise DescriptionError(f"No outfit table at {resolved}; run scripts/build_dataset.py.")
    frame = pd.read_parquet(resolved).set_index("image_id", drop=False)
    duplicated = int(frame.index.duplicated().sum())
    if duplicated:
        raise DescriptionError(f"Outfit table at {resolved} repeats {duplicated} image ids.")
    return frame


def _slots_from_mask(
    mask: np.ndarray, labels: dict[GarmentSlot, frozenset[int]]
) -> list[GarmentSlot]:
    """Return the slots owning at least one mask pixel, in slot declaration order."""
    return [slot for slot, _ in mask_skeleton(mask, labels)]


def _slots_from_texture(row: pd.Series) -> list[GarmentSlot]:
    """Return the slots whose texture columns are annotated, in slot declaration order."""
    return [
        slot
        for slot in GarmentSlot
        if slot in _TEXTURE_COLUMNS
        and any(row[column] != _NA for column in _TEXTURE_COLUMNS[slot])
    ]


def _garment(slot: GarmentSlot, row: pd.Series) -> Garment:
    """Build one garment from the texture columns that describe its slot."""
    columns = _TEXTURE_COLUMNS.get(slot)
    if columns is None:
        return Garment(slot=slot, category=SLOT_CATEGORIES[slot])
    return Garment(
        slot=slot,
        category=SLOT_CATEGORIES[slot],
        fabric=Fabric(row[columns[0]]),
        pattern=Pattern(row[columns[1]]),
    )


def _apply_shape(garments: list[Garment], row: pd.Series) -> None:
    """Attach the per-image shape annotation to the garments it describes."""
    by_slot = {garment.slot: garment for garment in garments}
    sleeved = next((by_slot[slot] for slot in _SLEEVED if slot in by_slot), None)
    if sleeved is not None:
        sleeved.sleeve_length = SleeveLength(row["sleeve_length"])
        sleeved.neckline = Neckline(row["neckline"])
    hemmed = next((by_slot[slot] for slot in _HEMMED if slot in by_slot), None)
    if hemmed is not None:
        hemmed.length = LowerLength(row["lower_length"])


def describe_image(
    image_id: str,
    row: pd.Series,
    *,
    config_path: Path | None = None,
) -> OutfitDescription:
    """Return the ground-truth description of one staged photograph."""
    labels = slot_labels(load_perception_config(config_path))
    picture = load_image(photo_path(image_id))
    keep = frozenset().union(*labels.values())
    mask = load_mask(image_id, (picture.height, picture.width), keep)

    slots = _slots_from_mask(mask, labels) if mask is not None else _slots_from_texture(row)
    if not slots:
        raise DescriptionError(f"No garment slot is populated for {image_id!r}.")

    garments = [_garment(slot, row) for slot in slots]
    if bool(row["has_shape"]):
        _apply_shape(garments, row)
    apply_palette(garments, picture, mask, labels)

    return OutfitDescription(
        image_id=image_id,
        image_path=paths.export_path(photo_path(image_id)),
        garments=garments,
        caption=str(row["caption"]),
        provenance=Provenance.gt,
        source_model=SOURCE_MODEL,
    )


def describe_split(
    split: str,
    *,
    limit: int | None = None,
    config_path: Path | None = None,
    table: pd.DataFrame | None = None,
) -> Iterator[OutfitDescription]:
    """Yield ground-truth descriptions for one frozen split, in sorted image id order."""
    frame = load_outfit_table() if table is None else table
    image_ids: Sequence[str] = sorted(load_split(split))
    if limit is not None:
        image_ids = image_ids[:limit]
    require(report_for_ids(image_ids, label=split))

    missing = [name for name in image_ids if name not in frame.index]
    if missing:
        raise DescriptionError(
            f"{len(missing)} ids in split {split!r} are absent from the outfit table; "
            f"first: {', '.join(missing[:3])}."
        )
    for image_id in image_ids:
        yield describe_image(image_id, frame.loc[image_id], config_path=config_path)
