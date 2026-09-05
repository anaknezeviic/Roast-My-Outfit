"""Mask and palette mechanics shared by every perception model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rmo import paths
from rmo.schemas import ColorName, Garment, GarmentSlot
from rmo.scoring.palette import PaletteEntry, extract_palette

log = logging.getLogger(__name__)

__all__ = [
    "HEAD_PROJECTION",
    "SLOT_CATEGORIES",
    "SLOT_PRIORITY",
    "apply_palette",
    "keep_labels",
    "load_mask",
    "mask_path",
    "mask_skeleton",
    "slot_labels",
]

_MASK_SUFFIX = "_segm"

SLOT_PRIORITY: tuple[GarmentSlot, ...] = (
    GarmentSlot.dress,
    GarmentSlot.romper,
    GarmentSlot.outer,
    GarmentSlot.lower,
    GarmentSlot.upper,
    GarmentSlot.footwear,
    GarmentSlot.bag,
    GarmentSlot.headwear,
    GarmentSlot.neckwear,
    GarmentSlot.socks,
    GarmentSlot.belt,
    GarmentSlot.gloves,
    GarmentSlot.eyewear,
    GarmentSlot.jewelry,
    GarmentSlot.other,
)

SLOT_CATEGORIES: dict[GarmentSlot, str] = {
    GarmentSlot.upper: "top",
    GarmentSlot.outer: "outerwear",
    GarmentSlot.lower: "bottom",
    GarmentSlot.dress: "dress",
    GarmentSlot.romper: "romper",
    GarmentSlot.footwear: "footwear",
    GarmentSlot.headwear: "headwear",
    GarmentSlot.neckwear: "neckwear",
    GarmentSlot.eyewear: "eyewear",
    GarmentSlot.bag: "bag",
    GarmentSlot.belt: "belt",
    GarmentSlot.socks: "socks",
    GarmentSlot.gloves: "gloves",
    GarmentSlot.jewelry: "jewelry",
    GarmentSlot.other: "unknown",
}

HEAD_PROJECTION: dict[str, tuple[str, tuple[GarmentSlot, ...]]] = {
    "upper_fabric": ("fabric", (GarmentSlot.upper,)),
    "lower_fabric": ("fabric", (GarmentSlot.lower,)),
    "outer_fabric": ("fabric", (GarmentSlot.outer,)),
    "upper_pattern": ("pattern", (GarmentSlot.upper,)),
    "lower_pattern": ("pattern", (GarmentSlot.lower,)),
    "outer_pattern": ("pattern", (GarmentSlot.outer,)),
    "sleeve_length": ("sleeve_length", (GarmentSlot.upper, GarmentSlot.outer)),
    "lower_length": ("length", (GarmentSlot.lower,)),
    "neckline": ("neckline", (GarmentSlot.upper, GarmentSlot.outer)),
}


def keep_labels(array: np.ndarray, labels: frozenset[int]) -> np.ndarray:
    """Return ``array`` with every label outside ``labels`` set to zero."""
    return np.where(np.isin(array, list(labels)), array, 0)


def mask_path(image_id: str) -> Path | None:
    """Return the parsing mask file for ``image_id``, or ``None`` when none is staged."""
    directory = paths.parsing_dir()
    return next(
        (
            candidate
            for candidate in (
                directory / f"{image_id}{_MASK_SUFFIX}.png",
                directory / f"{image_id}.png",
            )
            if candidate.is_file()
        ),
        None,
    )


def _measure(garment: Garment, entry: PaletteEntry, area_fraction: float | None) -> None:
    """Record a measured palette entry on a garment, keeping any colour the model stated."""
    garment.color_lab = entry.lab
    garment.color_lab_source = entry.source
    garment.area_fraction = area_fraction
    if garment.color is ColorName.unknown:
        garment.color = entry.name


def load_mask(image_id: str, size: tuple[int, int], keep: frozenset[int]) -> np.ndarray | None:
    """Return the parsing label map for ``image_id``, or ``None`` when it is unusable."""
    path = mask_path(image_id)
    if path is None:
        return None

    with Image.open(path) as handle:
        array = np.asarray(handle)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != size:
        log.warning("parsing mask for %s is %s, expected %s", image_id, array.shape, size)
        return None

    selected = keep_labels(array, keep)
    if not selected.any():
        return None
    return selected


def slot_labels(config: dict[str, Any]) -> dict[GarmentSlot, frozenset[int]]:
    """Return the parsing label ids that belong to each garment slot."""
    mask_labels = config.get("mask_labels", {})
    return {slot: frozenset(mask_labels.get(slot.value, ())) for slot in GarmentSlot}


def mask_skeleton(
    mask: np.ndarray, labels: Mapping[GarmentSlot, frozenset[int]]
) -> list[tuple[GarmentSlot, int]]:
    """Return the dominant parsing class of every slot the mask claims, in declaration order."""
    skeleton: list[tuple[GarmentSlot, int]] = []
    for slot in GarmentSlot:
        counts = {
            class_id: int(np.count_nonzero(mask == class_id))
            for class_id in sorted(labels.get(slot, frozenset()))
        }
        if not any(counts.values()):
            continue
        skeleton.append((slot, max(counts, key=lambda class_id: (counts[class_id], -class_id))))
    return skeleton


def apply_palette(
    garments: list[Garment],
    image: Image.Image,
    mask: np.ndarray | None,
    labels: dict[GarmentSlot, frozenset[int]],
) -> None:
    """Fill measured colour on each garment from the parsing mask, or the whole image."""
    if mask is None:
        entries = extract_palette(image, None, n_colors=1)
        if not entries:
            return
        target = min(
            enumerate(garments),
            key=lambda item: (SLOT_PRIORITY.index(item[1].slot), item[0]),
        )[1]
        _measure(target, entries[0], None)
        return

    denominator = int(np.count_nonzero(mask))
    for garment in garments:
        wanted = labels[garment.slot]
        if not wanted:
            continue
        selected = keep_labels(mask, wanted)
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        entries = extract_palette(image, selected, n_colors=1)
        if not entries:
            continue
        _measure(garment, entries[0], count / denominator)
