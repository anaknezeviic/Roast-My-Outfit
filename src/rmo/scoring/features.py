"""Fixed-width outfit feature vectors for pair construction and inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from rmo.config import ConfigError, load_scoring_config
from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
    SleeveLength,
)
from rmo.scoring.color import (
    RelationSettings,
    chroma,
    chroma_contrast,
    distinct_hue_count,
    lightness_contrast,
    neutral_count,
    pairwise_hue_deltas,
    resolved_lab,
    triadic_groups,
    violates_three_color_rule,
)
from rmo.scoring.rules import (
    FormalityTable,
    SeasonTable,
    cut_exposure,
    fabric_warmth,
    formality_level,
    largest_measured,
    rule_settings,
)

__all__ = [
    "MAX_MISSINGNESS_AUC",
    "ContractMismatch",
    "FeatureSpec",
    "build_spec",
    "describe_to_features",
    "feature_contract",
    "feature_names",
    "select_slots",
    "verify_contract",
]

MAX_MISSINGNESS_AUC = 0.6

_SEPARATOR = "__"
_SENTINEL = 0.0
_DTYPE = np.dtype(np.float64)
_SELECTION: dict[str, str] = {"strategy": "largest_measured_area", "tie_break": "ref"}

_ONE_HOT: dict[str, type[Enum]] = {
    "color": ColorName,
    "pattern": Pattern,
    "fabric": Fabric,
    "sleeve_length": SleeveLength,
    "length": LowerLength,
    "neckline": Neckline,
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable description of one feature vector layout."""

    version: str
    slots: tuple[GarmentSlot, ...]
    blocks: tuple[str, ...]
    relations: RelationSettings
    formality: FormalityTable
    season: SeasonTable
    names: tuple[str, ...]
    indicator_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Context:
    """One outfit reduced to the selected garments every block handler reads."""

    spec: FeatureSpec
    selected: Mapping[GarmentSlot, Garment | None]
    present: tuple[Garment, ...]
    n_ignored: int


@dataclass(frozen=True, slots=True)
class _Handler:
    """The columns one block contributes and the values it produces for an outfit."""

    columns: Callable[[FeatureSpec], tuple[tuple[str, bool], ...]]
    values: Callable[[_Context], list[float]]


def _one_hot(field: str, member: Enum | None) -> list[float]:
    """Return the one-hot block of an enum field, all zero when no member applies."""
    return [1.0 if candidate is member else 0.0 for candidate in _ONE_HOT[field]]


def _numeric(garment: Garment | None, value: float | None) -> list[float]:
    """Return a value column and its missing indicator, both zero for an absent slot."""
    if garment is None:
        return [_SENTINEL, 0.0]
    if value is None:
        return [_SENTINEL, 1.0]
    return [float(value), 0.0]


def _measured_lab(garment: Garment) -> tuple[float, float, float] | None:
    """Return the garment's own Lab when it is three finite numbers."""
    values = garment.color_lab
    if values is None or len(values) != 3 or not all(map(math.isfinite, values)):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _slot_block(
    group: tuple[tuple[str, bool], ...],
    values_of: Callable[[Garment | None, FeatureSpec], list[float]],
) -> _Handler:
    """Return a handler that repeats one column group over every slot of the spec."""

    def columns(spec: FeatureSpec) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (f"{slot.value}{_SEPARATOR}{suffix}", indicator)
            for slot in spec.slots
            for suffix, indicator in group
        )

    def values(context: _Context) -> list[float]:
        return [
            value
            for slot in context.spec.slots
            for value in values_of(context.selected[slot], context.spec)
        ]

    return _Handler(columns=columns, values=values)


def _enum_block(field: str, *, nullable: bool) -> _Handler:
    """Return the per-slot handler of one enum field, with an is-none indicator when nullable."""
    members = tuple(
        (f"{field}{_SEPARATOR}{member.value}", False) for member in _ONE_HOT[field]
    )
    group = members if not nullable else (*members, (f"{field}{_SEPARATOR}is_none", True))

    def values_of(garment: Garment | None, _spec: FeatureSpec) -> list[float]:
        """Return the one-hot columns of the field, plus its is-none flag when nullable."""
        member = None if garment is None else getattr(garment, field)
        block = _one_hot(field, member)
        if nullable:
            block.append(1.0 if garment is not None and member is None else 0.0)
        return block

    return _slot_block(group, values_of)


_PRESENCE_COLUMNS: tuple[tuple[str, bool], ...] = (("slot_missing", True),)

_COLOR_COLUMNS: tuple[tuple[str, bool], ...] = (
    *((f"color{_SEPARATOR}{member.value}", False) for member in ColorName),
    ("color_lab_l", False),
    ("color_lab_a", False),
    ("color_lab_b", False),
    ("color_lab_missing", True),
)

_AREA_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("area_fraction", False),
    ("area_missing", True),
)

_FORMALITY_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("formality_level", False),
    ("formality_missing", True),
)

_SEASON_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("fabric_warmth", False),
    ("fabric_warmth_missing", True),
    ("cut_exposure", False),
    ("cut_exposure_missing", True),
)


def _presence_values(garment: Garment | None, _spec: FeatureSpec) -> list[float]:
    """Return the missing flag of one slot."""
    return [1.0 if garment is None else 0.0]


def _color_values(garment: Garment | None, _spec: FeatureSpec) -> list[float]:
    """Return the colour one-hot block and the measured Lab of one slot."""
    block = _one_hot("color", None if garment is None else garment.color)
    lab = None if garment is None else _measured_lab(garment)
    if lab is None:
        return [*block, _SENTINEL, _SENTINEL, _SENTINEL, 0.0 if garment is None else 1.0]
    return [*block, *lab, 0.0]


def _area_values(garment: Garment | None, _spec: FeatureSpec) -> list[float]:
    """Return the measured area of one slot with its missing flag."""
    return _numeric(garment, None if garment is None else garment.area_fraction)


def _formality_values(garment: Garment | None, spec: FeatureSpec) -> list[float]:
    """Return the formality level of one slot with its missing flag."""
    level = (
        None
        if garment is None
        else formality_level(garment.category, table=spec.formality)
    )
    return _numeric(garment, None if level is None else float(level))


def _season_values(garment: Garment | None, spec: FeatureSpec) -> list[float]:
    """Return the fabric warmth and cut exposure of one slot with their missing flags."""
    warmth = None if garment is None else fabric_warmth(garment.fabric, table=spec.season)
    exposure = None if garment is None else cut_exposure(garment, table=spec.season)
    return _numeric(garment, warmth) + _numeric(garment, exposure)


def _aggregate(
    columns: tuple[tuple[str, bool], ...], values: Callable[[_Context], list[float]]
) -> _Handler:
    """Return a handler whose columns do not repeat over the slot list."""
    return _Handler(columns=lambda spec: columns, values=values)


_COLOR_AGGREGATE_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("color__hue_delta_mean", False),
    ("color__hue_delta_max", False),
    ("color__hue_delta_undefined", True),
    ("color__hue_pair_count", False),
    ("color__distinct_hue_families", False),
    ("color__chroma_contrast", False),
    ("color__chroma_contrast_undefined", True),
    ("color__lightness_contrast", False),
    ("color__lightness_contrast_undefined", True),
    ("color__neutral_count", False),
    ("color__three_color_rule_violated", True),
    ("color__triadic_group_count", False),
)

_FORMALITY_AGGREGATE_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("formality__level_spread", False),
    ("formality__level_mean", False),
    ("formality__mapped_count", False),
    ("formality__undefined", True),
)

_SEASON_AGGREGATE_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("season__max_fabric_warmth", False),
    ("season__max_cut_exposure", False),
    ("season__warmth_spread", False),
    ("season__undefined", True),
)

_PROPORTION_AGGREGATE_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("proportion__measured_area_sum", False),
    ("proportion__max_area", False),
    ("proportion__upper_lower_ratio", False),
    ("proportion__upper_lower_ratio_undefined", True),
    ("proportion__measured_count", False),
)

_COMPOSITION_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("composition__n_ignored", False),
    ("composition__n_selected", False),
)


def _color_aggregates(context: _Context) -> list[float]:
    """Return the outfit-wide colour relation columns."""
    garments = context.present
    settings = context.spec.relations
    deltas = pairwise_hue_deltas(garments, settings=settings)
    labs = [lab for lab in map(resolved_lab, garments) if lab is not None]
    chromatic = sum(chroma(lab) >= settings.achromatic_chroma for lab in labs)
    return [
        sum(deltas) / len(deltas) if deltas else _SENTINEL,
        max(deltas) if deltas else _SENTINEL,
        0.0 if deltas else 1.0,
        float(len(deltas)),
        float(distinct_hue_count(garments, settings=settings)),
        chroma_contrast(garments, settings=settings),
        0.0 if chromatic > 1 else 1.0,
        lightness_contrast(garments),
        0.0 if len(labs) > 1 else 1.0,
        float(neutral_count(garments)),
        float(violates_three_color_rule(garments, settings=settings)),
        float(len(triadic_groups(garments, settings=settings))),
    ]


def _formality_aggregates(context: _Context) -> list[float]:
    """Return the outfit-wide formality columns."""
    table = context.spec.formality
    levels = [
        level
        for garment in context.present
        if (level := formality_level(garment.category, table=table)) is not None
    ]
    if not levels:
        return [_SENTINEL, _SENTINEL, 0.0, 1.0]
    return [
        float(max(levels) - min(levels)),
        sum(levels) / len(levels),
        float(len(levels)),
        0.0,
    ]


def _season_aggregates(context: _Context) -> list[float]:
    """Return the outfit-wide season columns."""
    table = context.spec.season
    warmths = [
        value
        for garment in context.present
        if (value := fabric_warmth(garment.fabric, table=table)) is not None
    ]
    exposures = [
        value
        for garment in context.present
        if (value := cut_exposure(garment, table=table)) is not None
    ]
    return [
        max(warmths) if warmths else _SENTINEL,
        max(exposures) if exposures else _SENTINEL,
        max(warmths) - min(warmths) if len(warmths) > 1 else _SENTINEL,
        0.0 if warmths or exposures else 1.0,
    ]


def _upper_lower_ratio(selected: Mapping[GarmentSlot, Garment | None]) -> float | None:
    """Return the selected upper area over the selected lower area when both are measured."""
    upper = selected.get(GarmentSlot.upper)
    lower = selected.get(GarmentSlot.lower)
    if upper is None or lower is None:
        return None
    above, below = upper.area_fraction, lower.area_fraction
    if above is None or below is None or below <= 0.0:
        return None
    return above / below


def _proportion_aggregates(context: _Context) -> list[float]:
    """Return the outfit-wide area columns."""
    areas = [
        garment.area_fraction
        for garment in context.present
        if garment.area_fraction is not None
    ]
    ratio = _upper_lower_ratio(context.selected)
    return [
        float(sum(areas)),
        float(max(areas)) if areas else _SENTINEL,
        _SENTINEL if ratio is None else ratio,
        1.0 if ratio is None else 0.0,
        float(len(areas)),
    ]


def _composition(context: _Context) -> list[float]:
    """Return the counts of ignored and selected garments."""
    return [float(context.n_ignored), float(len(context.present))]


_HANDLERS: dict[str, _Handler] = {
    "slot_presence": _slot_block(_PRESENCE_COLUMNS, _presence_values),
    "slot_color": _slot_block(_COLOR_COLUMNS, _color_values),
    "slot_pattern": _enum_block("pattern", nullable=False),
    "slot_fabric": _enum_block("fabric", nullable=False),
    "slot_sleeve_length": _enum_block("sleeve_length", nullable=True),
    "slot_length": _enum_block("length", nullable=True),
    "slot_neckline": _enum_block("neckline", nullable=True),
    "slot_area": _slot_block(_AREA_COLUMNS, _area_values),
    "slot_formality": _slot_block(_FORMALITY_COLUMNS, _formality_values),
    "slot_season": _slot_block(_SEASON_COLUMNS, _season_values),
    "color_aggregates": _aggregate(_COLOR_AGGREGATE_COLUMNS, _color_aggregates),
    "formality_aggregates": _aggregate(
        _FORMALITY_AGGREGATE_COLUMNS, _formality_aggregates
    ),
    "season_aggregates": _aggregate(_SEASON_AGGREGATE_COLUMNS, _season_aggregates),
    "proportion_aggregates": _aggregate(
        _PROPORTION_AGGREGATE_COLUMNS, _proportion_aggregates
    ),
    "composition": _aggregate(_COMPOSITION_COLUMNS, _composition),
}


def _configured_slots(section: Mapping[str, Any]) -> tuple[GarmentSlot, ...]:
    """Return the ordered feature slots of the configuration."""
    raw = section.get("slots")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("features.slots must be a non-empty list.")

    slots: list[GarmentSlot] = []
    for name in raw:
        try:
            slot = GarmentSlot(name)
        except ValueError as exc:
            raise ConfigError(
                f"features.slots entry {name!r} is not a garment slot."
            ) from exc
        if slot in slots:
            raise ConfigError(f"features.slots repeats {name!r}.")
        slots.append(slot)
    return tuple(slots)


def _configured_blocks(section: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered feature blocks of the configuration."""
    raw = section.get("blocks")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("features.blocks must be a non-empty list.")

    blocks: list[str] = []
    for name in raw:
        if not isinstance(name, str) or name not in _HANDLERS:
            raise ConfigError(f"features.blocks entry {name!r} has no handler.")
        if name in blocks:
            raise ConfigError(f"features.blocks repeats {name!r}.")
        blocks.append(name)
    return tuple(blocks)


def build_spec(config: dict[str, Any] | None = None) -> FeatureSpec:
    """Return the immutable feature layout described by the scoring configuration."""
    resolved = load_scoring_config() if config is None else config
    settings = rule_settings(resolved)
    section = resolved.get("features")
    if not isinstance(section, dict):
        raise ConfigError("Scoring configuration has no features section.")

    version = section.get("version")
    if not isinstance(version, str) or not version:
        raise ConfigError(f"features.version must be a non-empty string; got {version!r}.")

    draft = FeatureSpec(
        version=version,
        slots=_configured_slots(section),
        blocks=_configured_blocks(section),
        relations=settings.relations,
        formality=settings.formality,
        season=settings.season,
        names=(),
        indicator_names=frozenset(),
    )
    columns = [
        column for block in draft.blocks for column in _HANDLERS[block].columns(draft)
    ]

    seen: set[str] = set()
    for name, _ in columns:
        if name in seen:
            raise ConfigError(f"features blocks produce duplicate column {name!r}.")
        seen.add(name)

    return replace(
        draft,
        names=tuple(name for name, _ in columns),
        indicator_names=frozenset(name for name, indicator in columns if indicator),
    )


def feature_names(spec: FeatureSpec) -> list[str]:
    """Return the ordered column names of a feature specification."""
    return list(spec.names)


def select_slots(
    description: OutfitDescription, spec: FeatureSpec
) -> dict[GarmentSlot, Garment | None]:
    """Return the representative garment of every feature slot, ``None`` when empty."""
    return {slot: largest_measured(description.by_slot(slot)) for slot in spec.slots}


def describe_to_features(description: OutfitDescription, spec: FeatureSpec) -> np.ndarray:
    """Return the fixed-width feature vector of one outfit description."""
    selected = select_slots(description, spec)
    in_scope = set(spec.slots)
    context = _Context(
        spec=spec,
        selected=selected,
        present=tuple(
            garment for slot in spec.slots if (garment := selected[slot]) is not None
        ),
        n_ignored=sum(garment.slot not in in_scope for garment in description.garments),
    )
    return np.array(
        [value for block in spec.blocks for value in _HANDLERS[block].values(context)],
        dtype=_DTYPE,
    )


def _member_scores(scores: Mapping[Any, float]) -> dict[str, float]:
    """Return an enum-keyed score table keyed by member value."""
    return {member.value: float(value) for member, value in scores.items()}


def _settings_payload(spec: FeatureSpec) -> dict[str, Any]:
    """Return the relation, formality and season settings as primitive mappings."""
    return {
        "relations": asdict(spec.relations),
        "formality": {
            "levels": {str(key): int(value) for key, value in spec.formality.levels.items()},
            "max_spread": spec.formality.max_spread,
            "major_spread": spec.formality.major_spread,
        },
        "season": {
            "fabric_warmth": _member_scores(spec.season.fabric_warmth),
            "sleeve_exposure": _member_scores(spec.season.sleeve_exposure),
            "length_exposure": _member_scores(spec.season.length_exposure),
            "heavy_warmth": spec.season.heavy_warmth,
            "summer_exposure": spec.season.summer_exposure,
            "max_fabric_warmth_spread": spec.season.max_fabric_warmth_spread,
        },
    }


def feature_contract(spec: FeatureSpec) -> dict[str, Any]:
    """Return the saveable contract of a feature specification with its digest."""
    contract: dict[str, Any] = {
        "feature_version": spec.version,
        "feature_names": list(spec.names),
        "slot_order": [slot.value for slot in spec.slots],
        "blocks": list(spec.blocks),
        "vocabularies": {
            field: [member.value for member in enum_type]
            for field, enum_type in _ONE_HOT.items()
        },
        "dtype": _DTYPE.name,
        "sentinel": _SENTINEL,
        "indicator_names": sorted(spec.indicator_names),
        "selection": dict(_SELECTION),
        "settings": _settings_payload(spec),
    }
    contract["spec_sha256"] = _contract_digest(contract)
    return contract


def _contract_digest(body: Mapping[str, Any]) -> str:
    """Return the digest of a contract body, ignoring any digest already in it."""
    payload = {key: value for key, value in body.items() if key != "spec_sha256"}
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ContractMismatch(ValueError):
    """Raised when a saved feature contract does not describe the current specification."""


def verify_contract(contract: Mapping[str, Any], spec: FeatureSpec) -> None:
    """Raise when ``contract`` was not produced by a specification equal to ``spec``."""
    if not isinstance(contract, Mapping):
        raise ContractMismatch("A feature contract must be a mapping.")

    stored = contract.get("spec_sha256")
    if not isinstance(stored, str) or not stored:
        raise ContractMismatch("The feature contract carries no spec_sha256.")
    if _contract_digest(contract) != stored:
        raise ContractMismatch(
            "The feature contract does not hash to its own spec_sha256, so either the "
            "digest or the body was edited after it was written."
        )

    current = feature_contract(spec)
    if stored == current["spec_sha256"]:
        return

    divergent = sorted(
        key
        for key in set(current) | set(contract)
        if key != "spec_sha256" and contract.get(key) != current.get(key)
    )
    saved_names = contract.get("feature_names")
    if isinstance(saved_names, list) and len(saved_names) != len(current["feature_names"]):
        raise ContractMismatch(
            f"The feature contract describes {len(saved_names)} features, but this "
            f"specification builds {len(current['feature_names'])}."
        )
    raise ContractMismatch(
        f"The feature contract disagrees on {', '.join(divergent)}."
    )
