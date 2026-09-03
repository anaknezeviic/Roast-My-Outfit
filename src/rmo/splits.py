"""Group-aware split loading.

Splits are keyed by garment group rather than by image, so every shot of one
product lands on the same side of the split.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rmo import paths

log = logging.getLogger(__name__)

__all__ = [
    "SPLIT_SEED",
    "SPLIT_NAMES",
    "group_key_for",
    "load_split",
    "assert_split_disjoint",
    "write_splits",
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


def _write_ids(path: Path, image_ids: set[str]) -> None:
    """Write sorted image stems with LF endings and a trailing newline."""
    content = "".join(f"{image_id}\n" for image_id in sorted(image_ids))
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content.encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_sha() -> str:
    """Return the current repository commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.repo_root(),
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _write_manifest(
    destination: Path,
    splits: dict[str, set[str]],
    split_groups: dict[str, set[str]],
) -> None:
    """Write split metadata and content hashes atomically."""
    manifest = {
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "n_groups": len(set().union(*split_groups.values())),
        "split_seed": SPLIT_SEED,
        "git_sha": _git_sha(),
        "sha256": {
            f"{name}.txt": hashlib.sha256(
                (destination / f"{name}.txt").read_bytes()
            ).hexdigest()
            for name in SPLIT_NAMES
        },
    }
    path = destination / "MANIFEST.json"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_splits(
    frame: pd.DataFrame,
    split_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Write deterministic grouped splits for full-body rows."""
    required = {"image_id", "gender", "category_from_filename", "is_full_body"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Outfit table is missing columns: {', '.join(sorted(missing_columns))}.")

    eligible = frame.loc[
        frame["is_full_body"],
        ["image_id", "gender", "category_from_filename"],
    ].copy()
    groups: list[str | None] = []
    for image_id in eligible["image_id"]:
        try:
            groups.append(group_key_for(image_id))
        except ValueError:
            groups.append(None)
    invalid = sum(group is None for group in groups)
    if invalid:
        log.warning("dropped %d full-body rows with invalid product filenames", invalid)
    eligible["group"] = groups
    eligible = eligible.dropna(subset=["group"])

    metadata = eligible[["gender", "category_from_filename"]]
    has_blank = metadata.apply(lambda column: column.astype(str).str.strip().eq("")).any().any()
    if metadata.isna().any().any() or has_blank:
        raise ValueError("Full-body rows have missing gender or category values.")
    if eligible["group"].nunique() < 7:
        raise ValueError("At least seven product groups are required to create splits.")

    eligible["stratum"] = (
        eligible["gender"].astype(str)
        + "|"
        + eligible["category_from_filename"].astype(str)
    )
    group_strata = eligible.groupby("group", sort=True)["stratum"].nunique()
    inconsistent = group_strata[group_strata != 1]
    if not inconsistent.empty:
        raise ValueError(f"Product group {inconsistent.index[0]!r} spans multiple strata.")

    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SPLIT_SEED)
    splits = {name: set() for name in SPLIT_NAMES}
    for fold, (_, held_out) in enumerate(
        splitter.split(eligible, eligible["stratum"], eligible["group"])
    ):
        name = "test" if fold == 0 else "val" if fold == 1 else "train"
        splits[name].update(eligible.iloc[held_out]["image_id"])

    split_groups = {
        name: {group_key_for(image_id) for image_id in image_ids}
        for name, image_ids in splits.items()
    }
    for first, second in itertools.combinations(SPLIT_NAMES, 2):
        if split_groups[first] & split_groups[second]:
            raise ValueError(f"Splits {first} and {second} share product groups.")

    destination = paths.ensure_dir(split_dir or paths.splits_dir())
    for name, image_ids in splits.items():
        _write_ids(destination / f"{name}.txt", image_ids)
    _write_manifest(destination, splits, split_groups)
    log.info(
        "wrote %d train, %d val and %d test image IDs to %s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        destination,
    )
    return splits
