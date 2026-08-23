"""Build the canonical outfit table from staged annotations and captions."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pandas as pd

from rmo import paths
from rmo.data.parse_annotations import SHAPE_COLUMNS, parse_label_dir
from rmo.splits import SPLIT_NAMES, write_splits

log = logging.getLogger(__name__)

__all__ = [
    "DatasetError",
    "build_dataset",
    "build_outfit_table",
    "load_captions",
    "main",
]

EXIT_BAD_INPUT = 2
EXIT_SPLITS_EXIST = 3

_IDENTITY = re.compile(
    r"^(?P<garment_id>(?P<gender>[^-]+)-(?P<category>.+)-id_\d{8})-(?P<shot>.+)$"
)
_CAPTION_NAMES = ("captions.json", "textual_descriptions.json")
_MASK_SUFFIX = "_segm"


class DatasetError(ValueError):
    """Raised when staged dataset inputs cannot form a canonical table."""


def _caption_text(value: object, image_id: str) -> str:
    """Normalize one caption value into text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [part.strip() for part in value if isinstance(part, str) and part.strip()]
        if len(parts) == len(value):
            return " ".join(parts)
    if isinstance(value, Mapping):
        for key in ("caption", "description", "text"):
            text = value.get(key)
            if isinstance(text, str):
                return text.strip()
    raise DatasetError(f"Caption for {image_id!r} is not text.")


def load_captions(path: Path) -> pd.DataFrame:
    """Load captions into a deterministic table keyed by image stem."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read captions from {path}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"Caption file {path} must contain a JSON object.")

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in payload.items():
        image_id = PurePosixPath(str(name)).stem
        if image_id in seen:
            raise DatasetError(f"Caption image {image_id!r} appears more than once.")
        seen.add(image_id)
        records.append((image_id, _caption_text(value, image_id)))
    return pd.DataFrame(records, columns=["image_id", "caption"]).sort_values(
        "image_id", ignore_index=True
    )


def _find_caption_file(raw_root: Path) -> Path:
    """Return the staged caption file."""
    for name in _CAPTION_NAMES:
        candidate = raw_root / name
        if candidate.is_file():
            return candidate
    matches = sorted(raw_root.glob("*caption*.json"))
    if len(matches) == 1:
        return matches[0]
    raise DatasetError(f"No caption JSON found in {raw_root}.")


def _identity_columns(image_ids: pd.Series) -> pd.DataFrame:
    """Derive filename identity columns."""
    extracted = image_ids.str.extract(_IDENTITY)
    return pd.DataFrame(
        {
            "garment_id": extracted["garment_id"],
            "gender": extracted["gender"].str.lower(),
            "category_from_filename": extracted["category"],
        }
    )


def build_outfit_table(
    label_dir: Path,
    caption_file: Path,
    parsing_dir: Path,
) -> pd.DataFrame:
    """Join staged metadata into one deterministic row per labeled image."""
    tables = parse_label_dir(label_dir)
    frame = tables["fabric"].merge(tables["pattern"], on="image_id", validate="one_to_one")
    frame = frame.merge(tables["shape"], on="image_id", how="left", validate="one_to_one")

    shape_columns = list(SHAPE_COLUMNS)
    frame["has_shape"] = frame[shape_columns].notna().all(axis="columns")
    for column in shape_columns:
        frame[column] = frame[column].fillna("na")

    identities = _identity_columns(frame["image_id"])
    valid_identity = identities.notna().all(axis="columns")
    dropped = int((~valid_identity).sum())
    if dropped:
        log.warning("dropped %d rows with invalid product filenames", dropped)
    frame = frame.loc[valid_identity].reset_index(drop=True)
    identities = identities.loc[valid_identity].reset_index(drop=True)
    for column in identities:
        frame[column] = identities[column]

    captions = load_captions(caption_file)
    frame = frame.merge(captions, on="image_id", how="left", validate="one_to_one")
    captioned = int(frame["caption"].notna().sum())
    if captioned == 0:
        raise DatasetError(f"No caption in {caption_file} names a labeled image.")
    if captioned < len(frame):
        log.info("%d of %d images carry no caption", len(frame) - captioned, len(frame))
    frame["caption"] = frame["caption"].fillna("")

    parsing_ids = (
        {
            candidate.stem.removesuffix(_MASK_SUFFIX)
            for candidate in parsing_dir.glob("*.png")
        }
        if parsing_dir.is_dir()
        else set()
    )
    frame["has_parsing"] = frame["image_id"].isin(parsing_ids)
    frame["is_full_body"] = frame["has_parsing"]

    trailing = ["caption", "has_shape", "has_parsing", "is_full_body"]
    leading = ["image_id", "garment_id", "gender", "category_from_filename"]
    columns = [
        *leading,
        *[
            column
            for column in frame.columns
            if column not in {*leading, *trailing}
        ],
        *trailing,
    ]
    return frame[columns].sort_values("image_id", ignore_index=True)


def build_dataset(
    raw_root: Path | None = None,
    output_path: Path | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build and atomically write ``outfits.parquet``."""
    raw_root = raw_root or paths.raw_dir()
    output_path = output_path or paths.processed_dir() / "outfits.parquet"
    frame = build_outfit_table(
        raw_root / "labels",
        _find_caption_file(raw_root),
        raw_root / "parsing",
    )
    if limit is not None:
        frame = frame.head(limit).reset_index(drop=True)
    paths.ensure_dir(output_path.parent)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    log.info("wrote %d outfits to %s", len(frame), output_path)
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Join staged annotations and captions into data/processed/outfits.parquet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Keep only the first N rows, for a smoke run.",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Also write the grouped train/val/test splits and their manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite splits that already exist. Every reported metric becomes stale.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.limit is not None and args.splits:
        parser.error("--splits needs the whole table; drop --limit")
    if args.force and not args.splits:
        parser.error("--force only applies to --splits")

    split_dir = paths.splits_dir()
    if args.splits and not args.force:
        existing = [
            name for name in SPLIT_NAMES if (split_dir / f"{name}.txt").is_file()
        ]
        if existing:
            log.error(
                "Splits already exist in %s and every committed metric is keyed to them. "
                "Pass --force only if you intend to invalidate those results.",
                split_dir,
            )
            return EXIT_SPLITS_EXIST

    try:
        frame = build_dataset(limit=args.limit)
        if args.splits:
            write_splits(frame)
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        return EXIT_BAD_INPUT

    return 0
