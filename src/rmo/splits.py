"""Group-aware split loading.

Splits are keyed by garment group rather than by image, so every shot of one
product lands on the same side of the split.
"""

from __future__ import annotations

import itertools
import re

from rmo import paths

__all__ = [
    "SPLIT_SEED",
    "SPLIT_NAMES",
    "group_key_for",
    "load_split",
    "assert_split_disjoint",
]

# Frozen; regenerating a split with a different seed invalidates the committed manifest.
SPLIT_SEED: int = 20260101

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

_GROUP_TOKEN = re.compile(r"id_\d{8}")


def group_key_for(image_id: str) -> str:
    """Return the product prefix that groups every shot of one garment."""
    match = _GROUP_TOKEN.search(image_id)
    if match is None:
        raise ValueError(f"Image id {image_id!r} carries no id_XXXXXXXX segment.")
    return image_id[: match.end()]


def load_split(name: str) -> set[str]:
    """Return the image ids listed in the named split file."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {', '.join(SPLIT_NAMES)}.")
    path = paths.splits_dir() / f"{name}.txt"
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def assert_split_disjoint() -> None:
    """Raise if any two splits share a garment group."""
    groups = {name: {group_key_for(image_id) for image_id in load_split(name)} for name in SPLIT_NAMES}
    for first, second in itertools.combinations(SPLIT_NAMES, 2):
        shared = groups[first] & groups[second]
        if shared:
            raise AssertionError(
                f"Splits {first} and {second} share {len(shared)} garment groups, "
                f"including {min(shared)}."
            )
