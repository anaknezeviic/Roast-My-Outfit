"""Build the canonical outfit table from staged annotations and captions."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pandas as pd

from rmo import paths
from rmo.data.parse_annotations import parse_label_dir

log = logging.getLogger(__name__)

__all__ = ["DatasetError", "build_dataset", "build_outfit_table", "load_captions"]

_IDENTITY = re.compile(
    r"^(?P<garment_id>(?P<gender>[^-]+)-(?P<category>.+)-id_\d{8})-(?P<shot>.+)$"
)
_CAPTION_NAMES = ("captions.json", "textual_descriptions.json")


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
    frame = tables["shape"].merge(tables["fabric"], on="image_id", validate="one_to_one")
    frame = frame.merge(tables["pattern"], on="image_id", validate="one_to_one")

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
    missing = frame.loc[frame["caption"].isna(), "image_id"].tolist()
    if missing:
        preview = ", ".join(repr(image_id) for image_id in missing[:3])
        raise DatasetError(f"Missing captions for {len(missing)} images: {preview}.")

    parsing_ids = (
        {candidate.stem for candidate in parsing_dir.glob("*.png")}
        if parsing_dir.is_dir()
        else set()
    )
    frame["has_parsing"] = frame["image_id"].isin(parsing_ids)
    frame["is_full_body"] = frame["has_parsing"]

    columns = [
        "image_id",
        "garment_id",
        "gender",
        "category_from_filename",
        *[
            column
            for column in frame.columns
            if column
            not in {
                "image_id",
                "garment_id",
                "gender",
                "category_from_filename",
                "caption",
                "has_parsing",
                "is_full_body",
            }
        ],
        "caption",
        "has_parsing",
        "is_full_body",
    ]
    return frame[columns].sort_values("image_id", ignore_index=True)


def build_dataset(
    raw_root: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build and atomically write ``outfits.parquet``."""
    raw_root = raw_root or paths.raw_dir()
    output_path = output_path or paths.data_root() / "processed" / "outfits.parquet"
    frame = build_outfit_table(
        raw_root / "labels",
        _find_caption_file(raw_root),
        raw_root / "parsing",
    )
    paths.ensure_dir(output_path.parent)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    log.info("wrote %d outfits to %s", len(frame), output_path)
    return frame
