"""Hand-authored outfit compatibility scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rmo.config import ConfigError, load_scoring_config
from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    Issue,
    IssueCode,
    IssueSeverity,
    LowerLength,
    OutfitDescription,
    OutfitScore,
    Pattern,
    SleeveLength,
    SubScores,
)
from rmo.scoring.base import ScoringModel
from rmo.scoring.color import (
    Relation,
    RelationSettings,
    distinct_hue_count,
    hue_pair_deltas,
    lightness_contrast,
    relation_of,
    relation_settings,
    resolved_lab,
    violates_three_color_rule,
)

__all__ = [
    "FormalityTable",
    "RuleScorer",
    "RuleSettings",
    "SeasonTable",
    "category_key",
    "cut_exposure",
    "fabric_warmth",
    "formality_level",
    "largest_measured",
    "rule_settings",
]

_MAX_LEVEL = 4

_SUBSCORE_FIELDS: tuple[str, ...] = tuple(SubScores.model_fields)

_PENALTY_KEYS: tuple[str, ...] = (
    "accessory_overload",
    "dominant_area",
    "fabric_mismatch",
    "formality_mismatch_per_level",
    "hue_clash",
    "low_contrast",
    "missing_footwear",
    "monochrome_flat",
    "pattern_clash",
    "season_mismatch",
    "too_many_colors",
    "upper_lower_ratio",
)

_MESSAGES: dict[str, str] = {
    "hue_clash": "The {a} and the {b} sit {delta:.0f} degrees apart in hue.",
    "too_many_colors": (
        "This outfit uses {families} separate hue families, above the limit of {limit}."
    ),
    "low_contrast": "The {a} and the {b} differ by {spread:.0f} points of lightness.",
    "monochrome_flat": (
        "The outfit stays on one hue family with a lightness spread of {spread:.0f} points."
    ),
    "pattern_clash": "This outfit has {count} patterned garments, above the limit of {limit}.",
    "formality_mismatch": "The {a} and the {b} sit {gap} formality levels apart.",
    "season_mismatch": "The {a} uses a heavy fabric while the {b} has an exposed cut.",
    "fabric_mismatch": "The {a} and the {b} use fabrics at opposite ends of the warmth range.",
    "dominant_area": "The {a} covers {share:.0f} percent of the frame.",
    "upper_lower_ratio": "The {a} covers {ratio:.1f} times the area of the {b}.",
    "missing_footwear": "No footwear is described for this outfit.",
    "accessory_overload": "This outfit has {count} accessories, above the limit of {limit}.",
    "other": "This description carries no usable colour, category or area evidence.",
}


@dataclass(frozen=True, slots=True)
class FormalityTable:
    """Formality level per normalised garment category, with its tolerated spread."""

    levels: Mapping[str, int]
    max_spread: int
    major_spread: int


@dataclass(frozen=True, slots=True)
class SeasonTable:
    """Fabric warmth and cut exposure lookups with the thresholds that compare them."""

    fabric_warmth: Mapping[Fabric, float]
    sleeve_exposure: Mapping[SleeveLength, float]
    length_exposure: Mapping[LowerLength, float]
    heavy_warmth: float
    summer_exposure: float
    max_fabric_warmth_spread: float


@dataclass(frozen=True, slots=True)
class RuleSettings:
    """Every table, threshold, weight and penalty the rule scorer reads."""

    relations: RelationSettings
    formality: FormalityTable
    season: SeasonTable
    min_lightness_contrast: float
    monochrome_lightness_contrast: float
    max_patterns: int
    accessory_slots: frozenset[GarmentSlot]
    dominant_slots: frozenset[GarmentSlot]
    core_slots: frozenset[GarmentSlot]
    max_accessories: int
    max_single_area: float
    max_upper_lower_ratio: float
    min_measured_areas: int
    min_garments_for_footwear: int
    weights: Mapping[str, float]
    penalties: Mapping[str, float]
    base: float
    unscorable: float
    max_issue_refs: int


def category_key(category: str) -> str:
    """Return the lookup form of a garment category."""
    return " ".join(category.replace("-", " ").lower().split())


def formality_level(category: str, *, table: FormalityTable) -> int | None:
    """Return the formality level of a category, ``None`` when it is unmapped."""
    return table.levels.get(category_key(category))


def fabric_warmth(fabric: Fabric, *, table: SeasonTable) -> float | None:
    """Return the warmth of a fabric, ``None`` when it carries no warmth evidence."""
    return table.fabric_warmth.get(fabric)


def cut_exposure(garment: Garment, *, table: SeasonTable) -> float | None:
    """Return the most exposed cut score of a garment, ``None`` when none is described."""
    sleeve = (
        None
        if garment.sleeve_length is None
        else table.sleeve_exposure.get(garment.sleeve_length)
    )
    hem = None if garment.length is None else table.length_exposure.get(garment.length)
    scored = [value for value in (sleeve, hem) if value is not None]
    return max(scored) if scored else None


def largest_measured(garments: Sequence[Garment]) -> Garment | None:
    """Return the garment with the largest measured area, ties broken by ref."""
    if not garments:
        return None
    return min(garments, key=_measurement_order)


def _measurement_order(garment: Garment) -> tuple[int, float, str]:
    """Return the sort key placing measured garments first and larger areas earlier."""
    if garment.area_fraction is None:
        return (1, 0.0, garment.ref)
    return (0, -garment.area_fraction, garment.ref)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one mapping section of the scoring configuration."""
    section = config.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Scoring configuration has no {name} section.")
    return section


def _number(
    section: Mapping[str, Any],
    path: str,
    key: str,
    *,
    low: float = 0.0,
    high: float = math.inf,
) -> float:
    """Return a finite option inside an inclusive range."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{path}.{key} must be a finite number; got {value!r}.")
    if not low <= value <= high:
        raise ConfigError(f"{path}.{key} must lie within [{low}, {high}]; got {value!r}.")
    return float(value)


def _count(section: Mapping[str, Any], path: str, key: str) -> int:
    """Return a strictly positive integer option."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{path}.{key} must be a positive integer; got {value!r}.")
    return value


def _level(path: str, key: str, value: Any) -> int:
    """Return an integer option inside the formality range."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_LEVEL:
        raise ConfigError(
            f"{path}.{key} must be an integer in [0, {_MAX_LEVEL}]; got {value!r}."
        )
    return value


def _formality_table(config: Mapping[str, Any]) -> FormalityTable:
    """Return the validated ``formality`` section."""
    section = _section(config, "formality")
    raw = section.get("levels")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("formality.levels must be a non-empty mapping.")

    levels: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key)
        if name != category_key(name):
            raise ConfigError(f"formality.levels key {name!r} is not normalised.")
        levels[name] = _level("formality.levels", name, value)

    max_spread = _level("formality", "max_spread", section.get("max_spread"))
    major_spread = _level("formality", "major_spread", section.get("major_spread"))
    if major_spread < max_spread:
        raise ConfigError("formality.major_spread must not be below formality.max_spread.")
    return FormalityTable(
        levels=MappingProxyType(levels), max_spread=max_spread, major_spread=major_spread
    )


def _enum_scores(
    section: Mapping[str, Any], key: str, enum_type: type[Enum]
) -> dict[Any, float]:
    """Return a validated mapping from enum members to finite scores."""
    raw = section.get(key)
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"season.{key} must be a non-empty mapping.")

    scores: dict[Any, float] = {}
    for name, value in raw.items():
        try:
            member = enum_type(name)
        except ValueError as exc:
            raise ConfigError(
                f"season.{key} key {name!r} is not a {enum_type.__name__} value."
            ) from exc
        if member.value == "na":
            raise ConfigError(f"season.{key} must not score the na member.")
        scores[member] = _number(raw, f"season.{key}", name, low=-math.inf)
    return scores


def _season_table(config: Mapping[str, Any]) -> SeasonTable:
    """Return the validated ``season`` section."""
    section = _section(config, "season")
    return SeasonTable(
        fabric_warmth=MappingProxyType(_enum_scores(section, "fabric_warmth", Fabric)),
        sleeve_exposure=MappingProxyType(
            _enum_scores(section, "sleeve_exposure", SleeveLength)
        ),
        length_exposure=MappingProxyType(
            _enum_scores(section, "length_exposure", LowerLength)
        ),
        heavy_warmth=_number(section, "season", "heavy_warmth", low=-math.inf),
        summer_exposure=_number(section, "season", "summer_exposure", low=-math.inf),
        max_fabric_warmth_spread=_number(section, "season", "max_fabric_warmth_spread"),
    )


def _slots(section: Mapping[str, Any], key: str) -> frozenset[GarmentSlot]:
    """Return a validated set of garment slots."""
    raw = section.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"proportion.{key} must be a non-empty list.")

    slots: set[GarmentSlot] = set()
    for name in raw:
        try:
            slots.add(GarmentSlot(name))
        except ValueError as exc:
            raise ConfigError(
                f"proportion.{key} entry {name!r} is not a garment slot."
            ) from exc
    return frozenset(slots)


def _weights(config: Mapping[str, Any]) -> dict[str, float]:
    """Return the validated ``weights`` section."""
    section = _section(config, "weights")
    if set(section) != set(_SUBSCORE_FIELDS):
        raise ConfigError(f"weights must cover exactly {', '.join(_SUBSCORE_FIELDS)}.")

    weights = {name: _number(section, "weights", name) for name in _SUBSCORE_FIELDS}
    if sum(weights.values()) <= 0.0:
        raise ConfigError("weights must not sum to zero.")
    return weights


def _penalties(config: Mapping[str, Any]) -> dict[str, float]:
    """Return the validated ``penalties`` section."""
    section = _section(config, "penalties")
    missing = sorted(set(_PENALTY_KEYS) - set(section))
    if missing:
        raise ConfigError(f"penalties is missing {', '.join(missing)}.")
    return {name: _number(section, "penalties", name) for name in _PENALTY_KEYS}


def rule_settings(config: dict[str, Any] | None = None) -> RuleSettings:
    """Return the validated rule tables of the scoring configuration."""
    resolved = load_scoring_config() if config is None else config
    harmony = _section(resolved, "harmony")
    proportion = _section(resolved, "proportion")
    defaults = _section(resolved, "defaults")

    max_single_area = _number(proportion, "proportion", "max_single_area", high=1)
    if max_single_area == 0.0:
        raise ConfigError("proportion.max_single_area must be greater than zero.")

    return RuleSettings(
        relations=relation_settings(resolved),
        formality=_formality_table(resolved),
        season=_season_table(resolved),
        min_lightness_contrast=_number(harmony, "harmony", "min_lightness_contrast"),
        monochrome_lightness_contrast=_number(
            harmony, "harmony", "monochrome_lightness_contrast"
        ),
        max_patterns=_count(harmony, "harmony", "max_patterns"),
        accessory_slots=_slots(proportion, "accessory_slots"),
        dominant_slots=_slots(proportion, "dominant_slots"),
        core_slots=_slots(proportion, "core_slots"),
        max_accessories=_count(proportion, "proportion", "max_accessories"),
        max_single_area=max_single_area,
        max_upper_lower_ratio=_number(
            proportion, "proportion", "max_upper_lower_ratio", low=1
        ),
        min_measured_areas=_count(proportion, "proportion", "min_measured_areas"),
        min_garments_for_footwear=_count(
            proportion, "proportion", "min_garments_for_footwear"
        ),
        weights=_weights(resolved),
        penalties=_penalties(resolved),
        base=_number(defaults, "defaults", "base", high=100),
        unscorable=_number(defaults, "defaults", "unscorable", high=100),
        max_issue_refs=_count(defaults, "defaults", "max_issue_refs"),
    )


def _clamp(value: float) -> float:
    """Return ``value`` limited to the scoring range."""
    return float(min(max(value, 0.0), 100.0))


def _describe(garment: Garment) -> str:
    """Return a short human label for one garment."""
    if garment.color is ColorName.unknown:
        return garment.category
    return f"{garment.color.value.replace('_', ' ')} {garment.category}"


def _refs(*garments: Garment) -> list[str]:
    """Return the refs of ``garments`` in order, without repeats."""
    listed: list[str] = []
    for garment in garments:
        if garment.ref not in listed:
            listed.append(garment.ref)
    return listed


def _has_evidence(garments: Sequence[Garment], settings: RuleSettings) -> bool:
    """Return whether any garment carries colour, category or area evidence."""
    return any(
        garment.area_fraction is not None
        or formality_level(garment.category, table=settings.formality) is not None
        or resolved_lab(garment) is not None
        for garment in garments
    )


def _color_harmony(
    garments: Sequence[Garment], settings: RuleSettings
) -> tuple[float, list[Issue]]:
    """Return the colour subscore and the colour issues of one outfit."""
    relations = settings.relations
    penalties = settings.penalties
    issues: list[Issue] = []
    penalty = 0.0

    by_ref = {garment.ref: garment for garment in garments}
    pairs = hue_pair_deltas(garments, settings=relations)
    hued = list(dict.fromkeys(ref for first, second, _ in pairs for ref in (first, second)))

    clash = next(
        (
            pair
            for pair in pairs
            if relation_of(pair[2], settings=relations) is Relation.complementary
        ),
        None,
    )
    if clash is not None:
        first, second = by_ref[clash[0]], by_ref[clash[1]]
        penalty += penalties["hue_clash"]
        issues.append(
            Issue(
                code=IssueCode.hue_clash,
                severity=IssueSeverity.major,
                message=_MESSAGES["hue_clash"].format(
                    a=_describe(first), b=_describe(second), delta=clash[2]
                ),
                garment_refs=_refs(first, second),
            )
        )

    refs = hued[: settings.max_issue_refs]
    families = distinct_hue_count(garments, settings=relations)
    if violates_three_color_rule(garments, settings=relations):
        penalty += penalties["too_many_colors"]
        issues.append(
            Issue(
                code=IssueCode.too_many_colors,
                severity=IssueSeverity.minor,
                message=_MESSAGES["too_many_colors"].format(
                    families=families, limit=relations.max_hue_families
                ),
                garment_refs=refs,
            )
        )

    lit = [
        (garment, lab[0])
        for garment in garments
        if (lab := resolved_lab(garment)) is not None
    ]
    spread = lightness_contrast(garments)
    monochrome = (
        len(lit) > 1
        and families == 1
        and (
            not pairs
            or max(delta for _, _, delta in pairs) <= relations.hue_separation
        )
        and spread < settings.monochrome_lightness_contrast
    )
    if monochrome:
        penalty += penalties["monochrome_flat"]
        issues.append(
            Issue(
                code=IssueCode.monochrome_flat,
                severity=IssueSeverity.minor,
                message=_MESSAGES["monochrome_flat"].format(spread=spread),
                garment_refs=refs,
            )
        )
    elif len(lit) > 1 and spread < settings.min_lightness_contrast:
        brightest = max(lit, key=lambda item: item[1])[0]
        darkest = min(lit, key=lambda item: item[1])[0]
        penalty += penalties["low_contrast"]
        issues.append(
            Issue(
                code=IssueCode.low_contrast,
                severity=IssueSeverity.minor,
                message=_MESSAGES["low_contrast"].format(
                    a=_describe(brightest), b=_describe(darkest), spread=spread
                ),
                garment_refs=_refs(brightest, darkest),
            )
        )

    patterned = [
        garment
        for garment in garments
        if garment.pattern not in (Pattern.pure_color, Pattern.na)
    ]
    if len(patterned) > settings.max_patterns:
        penalty += penalties["pattern_clash"]
        issues.append(
            Issue(
                code=IssueCode.pattern_clash,
                severity=IssueSeverity.minor,
                message=_MESSAGES["pattern_clash"].format(
                    count=len(patterned), limit=settings.max_patterns
                ),
                garment_refs=_refs(*patterned)[: settings.max_issue_refs],
            )
        )

    return (_clamp(settings.base - penalty), issues)


def _formality_consistency(
    garments: Sequence[Garment], settings: RuleSettings
) -> tuple[float, list[Issue]]:
    """Return the formality subscore and the formality issues of one outfit."""
    table = settings.formality
    mapped = [
        (garment, level)
        for garment in garments
        if (level := formality_level(garment.category, table=table)) is not None
    ]
    if len(mapped) < 2:
        return (_clamp(settings.base), [])

    highest, top = max(mapped, key=lambda item: item[1])
    lowest, bottom = min(mapped, key=lambda item: item[1])
    gap = top - bottom
    if gap <= table.max_spread:
        return (_clamp(settings.base), [])

    penalty = settings.penalties["formality_mismatch_per_level"] * (gap - table.max_spread)
    issue = Issue(
        code=IssueCode.formality_mismatch,
        severity=(
            IssueSeverity.major if gap >= table.major_spread else IssueSeverity.minor
        ),
        message=_MESSAGES["formality_mismatch"].format(
            a=_describe(highest), b=_describe(lowest), gap=gap
        ),
        garment_refs=_refs(highest, lowest),
    )
    return (_clamp(settings.base - penalty), [issue])


def _seasonality(
    garments: Sequence[Garment], settings: RuleSettings
) -> tuple[float, list[Issue]]:
    """Return the season subscore and the season issues of one outfit."""
    table = settings.season
    penalties = settings.penalties
    issues: list[Issue] = []
    penalty = 0.0

    warmths = [
        (garment, value)
        for garment in garments
        if (value := fabric_warmth(garment.fabric, table=table)) is not None
    ]
    exposures = [
        (garment, value)
        for garment in garments
        if (value := cut_exposure(garment, table=table)) is not None
    ]

    if warmths and exposures:
        heaviest, warmth = max(warmths, key=lambda item: item[1])
        barest, exposure = max(exposures, key=lambda item: item[1])
        if warmth >= table.heavy_warmth and exposure >= table.summer_exposure:
            penalty += penalties["season_mismatch"]
            issues.append(
                Issue(
                    code=IssueCode.season_mismatch,
                    severity=IssueSeverity.major,
                    message=_MESSAGES["season_mismatch"].format(
                        a=_describe(heaviest), b=_describe(barest)
                    ),
                    garment_refs=_refs(heaviest, barest),
                )
            )

    if len(warmths) > 1:
        hottest, high = max(warmths, key=lambda item: item[1])
        coldest, low = min(warmths, key=lambda item: item[1])
        if high - low > table.max_fabric_warmth_spread:
            penalty += penalties["fabric_mismatch"]
            issues.append(
                Issue(
                    code=IssueCode.fabric_mismatch,
                    severity=IssueSeverity.minor,
                    message=_MESSAGES["fabric_mismatch"].format(
                        a=_describe(hottest), b=_describe(coldest)
                    ),
                    garment_refs=_refs(hottest, coldest),
                )
            )

    return (_clamp(settings.base - penalty), issues)


def _proportion(
    garments: Sequence[Garment], settings: RuleSettings
) -> tuple[float, list[Issue]]:
    """Return the proportion subscore and the proportion issues of one outfit."""
    penalties = settings.penalties
    issues: list[Issue] = []
    penalty = 0.0
    refs = [garment.ref for garment in garments][: settings.max_issue_refs]

    wears_footwear = any(garment.slot is GarmentSlot.footwear for garment in garments)
    covers_core = any(garment.slot in settings.core_slots for garment in garments)
    if (
        not wears_footwear
        and covers_core
        and len(garments) >= settings.min_garments_for_footwear
    ):
        penalty += penalties["missing_footwear"]
        issues.append(
            Issue(
                code=IssueCode.missing_footwear,
                severity=IssueSeverity.minor,
                message=_MESSAGES["missing_footwear"],
                garment_refs=refs,
            )
        )

    accessories = [
        garment for garment in garments if garment.slot in settings.accessory_slots
    ]
    if len(accessories) > settings.max_accessories:
        penalty += penalties["accessory_overload"]
        issues.append(
            Issue(
                code=IssueCode.accessory_overload,
                severity=IssueSeverity.minor,
                message=_MESSAGES["accessory_overload"].format(
                    count=len(accessories), limit=settings.max_accessories
                ),
                garment_refs=_refs(*accessories),
            )
        )

    measured = [garment for garment in garments if garment.area_fraction is not None]
    if len(measured) >= settings.min_measured_areas:
        biggest = largest_measured(
            [garment for garment in measured if garment.slot not in settings.dominant_slots]
        )
        share = None if biggest is None else biggest.area_fraction
        if biggest is not None and share is not None and share > settings.max_single_area:
            penalty += penalties["dominant_area"]
            issues.append(
                Issue(
                    code=IssueCode.proportion_imbalance,
                    severity=IssueSeverity.minor,
                    message=_MESSAGES["dominant_area"].format(
                        a=_describe(biggest), share=share * 100.0
                    ),
                    garment_refs=_refs(biggest),
                )
            )

    balance = _upper_lower_balance(measured, settings)
    if balance is not None:
        bigger, smaller, ratio = balance
        penalty += penalties["upper_lower_ratio"]
        issues.append(
            Issue(
                code=IssueCode.proportion_imbalance,
                severity=IssueSeverity.minor,
                message=_MESSAGES["upper_lower_ratio"].format(
                    a=_describe(bigger), b=_describe(smaller), ratio=ratio
                ),
                garment_refs=_refs(bigger, smaller),
            )
        )

    return (_clamp(settings.base - penalty), issues)


def _upper_lower_balance(
    measured: Sequence[Garment], settings: RuleSettings
) -> tuple[Garment, Garment, float] | None:
    """Return the unbalanced upper and lower representatives with their area ratio."""
    upper = largest_measured(
        [garment for garment in measured if garment.slot is GarmentSlot.upper]
    )
    lower = largest_measured(
        [garment for garment in measured if garment.slot is GarmentSlot.lower]
    )
    if upper is None or lower is None:
        return None

    above = upper.area_fraction
    below = lower.area_fraction
    if above is None or below is None:
        return None

    bigger, smaller = (upper, lower) if above >= below else (lower, upper)
    top, bottom = (above, below) if above >= below else (below, above)
    if bottom <= 0.0:
        return None

    ratio = top / bottom
    return (bigger, smaller, ratio) if ratio > settings.max_upper_lower_ratio else None


class RuleScorer(ScoringModel):
    """Score an outfit description against configured style heuristics."""

    name = "rule_scorer_v1"

    def __init__(self, config_path: Path | None = None) -> None:
        """Read and validate the scoring tables without touching any image."""
        self._settings = rule_settings(load_scoring_config(config_path))

    def score(self, description: OutfitDescription) -> OutfitScore:
        """Return the score for one outfit description."""
        settings = self._settings
        garments = description.garments

        if not _has_evidence(garments, settings):
            return self._unscorable(description, settings)

        parts = {
            "color_harmony": _color_harmony(garments, settings),
            "formality_consistency": _formality_consistency(garments, settings),
            "seasonality": _seasonality(garments, settings),
            "proportion": _proportion(garments, settings),
        }
        weights = settings.weights
        overall = sum(weights[name] * value for name, (value, _) in parts.items()) / sum(
            weights.values()
        )

        return OutfitScore(
            image_id=description.image_id,
            overall=_clamp(overall),
            subscores=SubScores(**{name: value for name, (value, _) in parts.items()}),
            issues=[issue for _, found in parts.values() for issue in found],
            provenance=description.provenance,
            source_model=self.name,
        )

    def _unscorable(
        self, description: OutfitDescription, settings: RuleSettings
    ) -> OutfitScore:
        """Return the neutral score used when a description carries no usable evidence."""
        neutral = _clamp(settings.unscorable)
        return OutfitScore(
            image_id=description.image_id,
            overall=neutral,
            subscores=SubScores(**{name: neutral for name in _SUBSCORE_FIELDS}),
            issues=[
                Issue(
                    code=IssueCode.other,
                    severity=IssueSeverity.info,
                    message=_MESSAGES["other"],
                    garment_refs=description.refs()[: settings.max_issue_refs],
                )
            ],
            provenance=description.provenance,
            source_model=self.name,
        )
