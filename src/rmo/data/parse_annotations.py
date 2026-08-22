"""Decode the three integer-coded label files into tidy per-image tables.

Filenames differ between mirrors, so files are identified by row width and name
rather than by a fixed filename. ``na`` is a real category meaning "not visible",
never a null.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "AnnotationError",
    "FABRIC_COLUMNS",
    "PATTERN_COLUMNS",
    "SHAPE_COLUMNS",
    "find_label_files",
    "parse_fabric",
    "parse_label_dir",
    "parse_pattern",
    "parse_shape",
]


class AnnotationError(ValueError):
    """Raised when a label file is malformed or inconsistent with its siblings."""


_Field = tuple[str, tuple[str, ...]]

_SLEEVE_LENGTH = ("sleeveless", "short", "medium", "long", "not_long", "na")
_LOWER_LENGTH = ("three_point", "medium_short", "three_quarter", "long", "na")
_SOCKS = ("no", "socks", "leggings", "na")
_YES_NO = ("no", "yes", "na")
_GLASSES = ("no", "eyeglasses", "sunglasses", "in_hand_or_clothes", "na")
_WAIST = ("no", "belt", "clothing", "hidden", "na")
_NECKLINE = ("v_shape", "square", "round", "standing", "lapel", "suspenders", "na")
# code 0 is "yes" here, the reverse of every other flag in this file
_CARDIGAN = ("yes", "no", "na")
_FABRIC = ("denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "na")
# "other" is code 5 and "color_block" code 6, which is not the order they are usually listed in
_PATTERN = ("floral", "graphic", "striped", "pure_color", "lattice", "other", "color_block", "na")

_SHAPE_FIELDS: tuple[_Field, ...] = (
    ("sleeve_length", _SLEEVE_LENGTH),
    ("lower_length", _LOWER_LENGTH),
    ("socks", _SOCKS),
    ("hat", _YES_NO),
    ("glasses", _GLASSES),
    ("neckwear", _YES_NO),
    ("wrist", _YES_NO),
    ("ring", _YES_NO),
    ("waist_accessory", _WAIST),
    ("neckline", _NECKLINE),
    ("cardigan", _CARDIGAN),
    ("navel_covered", _YES_NO),
)

_FABRIC_FIELDS: tuple[_Field, ...] = (
    ("upper_fabric", _FABRIC),
    ("lower_fabric", _FABRIC),
    ("outer_fabric", _FABRIC),
)

_PATTERN_FIELDS: tuple[_Field, ...] = (
    ("upper_pattern", _PATTERN),
    ("lower_pattern", _PATTERN),
    ("outer_pattern", _PATTERN),
)

SHAPE_COLUMNS = tuple(name for name, _ in _SHAPE_FIELDS)
FABRIC_COLUMNS = tuple(name for name, _ in _FABRIC_FIELDS)
PATTERN_COLUMNS = tuple(name for name, _ in _PATTERN_FIELDS)

_SHAPE_WIDTH = len(_SHAPE_FIELDS) + 1
_TEXTURE_WIDTH = len(_FABRIC_FIELDS) + 1
_FABRIC_TOKENS = ("fabric",)
_PATTERN_TOKENS = ("pattern", "color", "colour", "texture")
_KINDS = ("shape", "fabric", "pattern")


def _image_id(name: str) -> str:
    """Return the filename stem, which is the join key used by every later stage."""
    return PurePosixPath(name).stem


def _rows(path: Path, width: int) -> Iterator[tuple[int, str, list[str]]]:
    """Yield ``(line number, image name, codes)`` for every non-blank line."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != width:
                raise AnnotationError(
                    f"{path.name} line {number}: expected {width} fields, found {len(parts)}."
                )
            yield number, parts[0], parts[1:]


def _decode(path: Path, number: int, column: str, raw: str, vocabulary: Sequence[str]) -> str:
    """Translate one integer code into its vocabulary value."""
    try:
        code = int(raw)
    except ValueError:
        raise AnnotationError(
            f"{path.name} line {number}: {column} value {raw!r} is not an integer."
        ) from None
    if not 0 <= code < len(vocabulary):
        raise AnnotationError(
            f"{path.name} line {number}: {column} code {code} is outside 0..{len(vocabulary) - 1}."
        )
    return vocabulary[code]


def _parse(path: Path, fields: Sequence[_Field]) -> pd.DataFrame:
    """Decode ``path`` into one row per image, sorted by ``image_id``."""
    if not path.is_file():
        raise AnnotationError(f"Annotation file {path} does not exist.")

    columns = [name for name, _ in fields]
    seen: dict[str, int] = {}
    records: list[tuple[str, ...]] = []

    for number, name, codes in _rows(path, len(fields) + 1):
        image_id = _image_id(name)
        if image_id in seen:
            raise AnnotationError(
                f"{path.name} line {number}: image {image_id!r} already appeared on line {seen[image_id]}."
            )
        seen[image_id] = number
        decoded = tuple(
            _decode(path, number, column, raw, vocabulary)
            for (column, vocabulary), raw in zip(fields, codes)
        )
        records.append((image_id, *decoded))

    frame = pd.DataFrame(records, columns=["image_id", *columns])
    frame = frame.sort_values("image_id", ignore_index=True)
    for column, vocabulary in fields:
        frame[column] = pd.Categorical(frame[column], categories=vocabulary)
    return frame


def parse_shape(path: Path) -> pd.DataFrame:
    """Return one row per image carrying the twelve decoded shape attributes."""
    return _parse(path, _SHAPE_FIELDS)


def parse_fabric(path: Path) -> pd.DataFrame:
    """Return one row per image carrying upper, lower and outer fabric."""
    return _parse(path, _FABRIC_FIELDS)


def parse_pattern(path: Path) -> pd.DataFrame:
    """Return one row per image carrying upper, lower and outer pattern."""
    return _parse(path, _PATTERN_FIELDS)


_PARSERS = {"shape": parse_shape, "fabric": parse_fabric, "pattern": parse_pattern}


def _first_row_width(path: Path) -> int:
    """Return the field count of the first non-blank line, or 0 when there is none."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return len(line.split())
    return 0


def find_label_files(label_dir: Path) -> dict[str, Path]:
    """Map ``shape``, ``fabric`` and ``pattern`` onto the label files in ``label_dir``."""
    if not label_dir.is_dir():
        raise AnnotationError(f"Label directory {label_dir} does not exist.")

    found: dict[str, Path] = {}
    for candidate in sorted(label_dir.glob("*.txt")):
        width = _first_row_width(candidate)
        if width == _SHAPE_WIDTH:
            found.setdefault("shape", candidate)
        elif width == _TEXTURE_WIDTH:
            # fabric and pattern share a row width, so the name is the only discriminator
            lowered = candidate.name.lower()
            if any(token in lowered for token in _FABRIC_TOKENS):
                found.setdefault("fabric", candidate)
            elif any(token in lowered for token in _PATTERN_TOKENS):
                found.setdefault("pattern", candidate)

    missing = [kind for kind in _KINDS if kind not in found]
    if missing:
        raise AnnotationError(f"No {', '.join(missing)} annotation found in {label_dir}.")
    return found


def parse_label_dir(label_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse all three label files and confirm they describe the same images."""
    files = find_label_files(label_dir)
    tables = {kind: _PARSERS[kind](files[kind]) for kind in _KINDS}

    reference = set(tables["shape"]["image_id"])
    for kind in ("fabric", "pattern"):
        other = set(tables[kind]["image_id"])
        if other != reference:
            raise AnnotationError(
                f"{files[kind].name} covers {len(other)} images against "
                f"{len(reference)} in {files['shape'].name}: "
                f"{len(reference - other)} missing, {len(other - reference)} unexpected."
            )

    log.info("parsed %d images from %s", len(reference), label_dir)
    return tables
