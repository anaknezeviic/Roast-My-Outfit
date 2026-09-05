"""Colour-theory relations between the garments of one outfit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from rmo.config import ConfigError, load_scoring_config
from rmo.schemas import NEUTRAL_COLORS, ColorName, Garment
from rmo.scoring.palette import reference_lab

__all__ = [
    "Relation",
    "RelationSettings",
    "chroma",
    "chroma_contrast",
    "distinct_hue_count",
    "hue_angle",
    "hue_pair_deltas",
    "lightness_contrast",
    "neutral_count",
    "pairwise_hue_deltas",
    "relation_of",
    "relation_settings",
    "resolved_lab",
    "triadic_groups",
    "violates_three_color_rule",
]

_FULL_TURN = 360.0
_STRAIGHT = 180.0
_TRIAD_STEP = 120.0


class Relation(str, Enum):
    """Colour-theory relation carried by a hue delta."""

    complementary = "complementary"
    analogous = "analogous"
    triadic_candidate = "triadic_candidate"
    none = "none"


@dataclass(frozen=True, slots=True)
class RelationSettings:
    """Tolerances that turn hue geometry into named relations."""

    achromatic_chroma: float
    analogous_max: float
    complementary_tolerance: float
    triadic_tolerance: float
    hue_separation: float
    max_hue_families: int


def _tolerance(section: Mapping[str, Any], key: str, ceiling: float | None) -> float:
    """Return a finite non-negative option, bounded above when the key has a ceiling."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"color.{key} must be a finite number; got {value!r}.")
    if value < 0.0:
        raise ConfigError(f"color.{key} must not be negative; got {value!r}.")
    if ceiling is not None and value > ceiling:
        raise ConfigError(f"color.{key} must not exceed {ceiling}; got {value!r}.")
    return float(value)


def relation_settings(config: dict[str, Any] | None = None) -> RelationSettings:
    """Return the validated ``color`` section of the scoring configuration."""
    section = (load_scoring_config() if config is None else config).get("color")
    if not isinstance(section, dict):
        raise ConfigError("Scoring configuration has no color section.")

    families = section.get("max_hue_families")
    if isinstance(families, bool) or not isinstance(families, int) or families < 1:
        raise ConfigError(f"color.max_hue_families must be a positive integer; got {families!r}.")

    return RelationSettings(
        achromatic_chroma=_tolerance(section, "achromatic_chroma", None),
        analogous_max=_tolerance(section, "analogous_max", _STRAIGHT),
        complementary_tolerance=_tolerance(section, "complementary_tolerance", 90.0),
        triadic_tolerance=_tolerance(section, "triadic_tolerance", 60.0),
        hue_separation=_tolerance(section, "hue_separation", _STRAIGHT),
        max_hue_families=families,
    )


def _checked_lab(lab: Sequence[float]) -> tuple[float, float, float]:
    """Return ``lab`` as exactly three finite floats."""
    values = tuple(lab)
    if len(values) != 3 or not all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"Lab value must be three finite numbers; got {lab!r}.")
    return (float(values[0]), float(values[1]), float(values[2]))


def chroma(lab: Sequence[float]) -> float:
    """Return the CIELAB chroma of ``lab``."""
    _, a, b = _checked_lab(lab)
    return math.hypot(a, b)


def hue_angle(lab: Sequence[float], *, settings: RelationSettings) -> float | None:
    """Return the CIELAB hue angle in degrees, ``None`` below the achromatic threshold."""
    _, a, b = _checked_lab(lab)
    if math.hypot(a, b) < settings.achromatic_chroma:
        return None
    return math.degrees(math.atan2(b, a)) % _FULL_TURN


def resolved_lab(garment: Garment) -> tuple[float, float, float] | None:
    """Return the garment's measured Lab when it is usable, else its named reference."""
    measured = garment.color_lab
    if measured is not None and len(measured) == 3 and all(map(math.isfinite, measured)):
        return (float(measured[0]), float(measured[1]), float(measured[2]))
    return reference_lab(garment.color)


def _hue_carrying(
    garments: Sequence[Garment], settings: RelationSettings
) -> list[tuple[str, float]]:
    """Return the ref and hue angle of every garment whose colour carries a styling hue."""
    carried: list[tuple[str, float]] = []
    for garment in garments:
        if garment.color in NEUTRAL_COLORS or garment.color is ColorName.unknown:
            continue
        lab = resolved_lab(garment)
        if lab is None:
            continue
        angle = hue_angle(lab, settings=settings)
        if angle is not None:
            carried.append((garment.ref, angle))
    return carried


def _folded(first: float, second: float) -> float:
    """Return the circular distance between two hue angles, folded into ``[0, 180]``."""
    delta = abs(first - second) % _FULL_TURN
    return _FULL_TURN - delta if delta > _STRAIGHT else delta


def neutral_count(garments: Sequence[Garment]) -> int:
    """Return how many garments carry a neutral colour name."""
    return sum(garment.color in NEUTRAL_COLORS for garment in garments)


def hue_pair_deltas(
    garments: Sequence[Garment], *, settings: RelationSettings
) -> list[tuple[str, str, float]]:
    """Return the ref pair and hue delta of every pair of hue-carrying garments."""
    return [
        (first[0], second[0], _folded(first[1], second[1]))
        for first, second in combinations(_hue_carrying(garments, settings), 2)
    ]


def pairwise_hue_deltas(
    garments: Sequence[Garment], *, settings: RelationSettings
) -> list[float]:
    """Return the hue delta of every pair of hue-carrying garments."""
    return [delta for _, _, delta in hue_pair_deltas(garments, settings=settings)]


def relation_of(delta: float, *, settings: RelationSettings) -> Relation:
    """Return the relation a hue delta expresses."""
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(delta)
        or not 0.0 <= delta <= _STRAIGHT
    ):
        raise ValueError(f"Hue delta must be a finite number in [0, 180]; got {delta!r}.")
    if delta >= _STRAIGHT - settings.complementary_tolerance:
        return Relation.complementary
    if delta <= settings.analogous_max:
        return Relation.analogous
    if abs(delta - _TRIAD_STEP) <= settings.triadic_tolerance:
        return Relation.triadic_candidate
    return Relation.none


def triadic_groups(
    garments: Sequence[Garment], *, settings: RelationSettings
) -> list[tuple[str, str, str]]:
    """Return every triple of refs whose hue angles sit near an even three-way split."""
    groups: set[tuple[str, str, str]] = set()
    for triple in combinations(_hue_carrying(garments, settings), 3):
        angles = sorted(angle for _, angle in triple)
        gaps = (
            angles[1] - angles[0],
            angles[2] - angles[1],
            _FULL_TURN - (angles[2] - angles[0]),
        )
        if all(abs(gap - _TRIAD_STEP) <= settings.triadic_tolerance for gap in gaps):
            refs = sorted(ref for ref, _ in triple)
            groups.add((refs[0], refs[1], refs[2]))
    return sorted(groups)


def chroma_contrast(garments: Sequence[Garment], *, settings: RelationSettings) -> float:
    """Return the chroma spread across the garments that clear the achromatic threshold."""
    values = [
        value
        for lab in map(resolved_lab, garments)
        if lab is not None and (value := chroma(lab)) >= settings.achromatic_chroma
    ]
    return max(values) - min(values) if len(values) > 1 else 0.0


def lightness_contrast(garments: Sequence[Garment]) -> float:
    """Return the lightness spread across the garments with a resolved Lab."""
    values = [lab[0] for lab in map(resolved_lab, garments) if lab is not None]
    return max(values) - min(values) if len(values) > 1 else 0.0


def distinct_hue_count(garments: Sequence[Garment], *, settings: RelationSettings) -> int:
    """Return how many separated hue families the hue-carrying garments occupy."""
    angles = sorted(angle for _, angle in _hue_carrying(garments, settings))
    if not angles:
        return 0
    gaps = [later - earlier for earlier, later in zip(angles, angles[1:])]
    gaps.append(_FULL_TURN - (angles[-1] - angles[0]))
    return max(sum(gap > settings.hue_separation for gap in gaps), 1)


def violates_three_color_rule(
    garments: Sequence[Garment], *, settings: RelationSettings
) -> bool:
    """Return whether the outfit spreads over more hue families than the config allows."""
    return distinct_hue_count(garments, settings=settings) > settings.max_hue_families
