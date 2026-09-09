"""Resolve raw DeepFashion regional labels into the canonical garment schema.

DeepFashion-MultiModal stores fabric/pattern annotations by body region
(``upper_*``, ``lower_*``, ``outer_*``), while RMO models dresses and rompers as
one physical garment.  This module contains the single deterministic rule for
bridging those two representations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from rmo.schemas import GarmentSlot

__all__ = [
    "NA_VALUE",
    "ONE_PIECE_SLOTS",
    "RegionalResolution",
    "resolve_regional_value",
    "texture_value",
]

NA_VALUE = "na"
ONE_PIECE_SLOTS: tuple[GarmentSlot, ...] = (GarmentSlot.dress, GarmentSlot.romper)

ResolutionKind = Literal["both_na", "upper_only", "lower_only", "same", "conflict"]


@dataclass(frozen=True, slots=True)
class RegionalResolution:
    """Canonical result of reconciling upper/lower labels for one garment."""

    value: str
    kind: ResolutionKind

    @property
    def conflicted(self) -> bool:
        """Return whether two visible regional labels disagree."""
        return self.kind == "conflict"


_DIRECT_COLUMNS: dict[GarmentSlot, dict[str, str]] = {
    GarmentSlot.upper: {"fabric": "upper_fabric", "pattern": "upper_pattern"},
    GarmentSlot.outer: {"fabric": "outer_fabric", "pattern": "outer_pattern"},
    GarmentSlot.lower: {"fabric": "lower_fabric", "pattern": "lower_pattern"},
}


def resolve_regional_value(upper: object, lower: object) -> RegionalResolution:
    """Collapse upper/lower regional labels without inventing a false label.

    Equal labels are preserved.  When only one side is annotated, that visible
    value is preserved.  If both visible regions carry different labels, the
    current one-value-per-garment schema cannot represent the annotation
    faithfully, so ``na`` is returned with ``kind='conflict'``.  Training code
    can then omit that field instead of supervising a fabricated choice.
    """
    upper_value = str(upper)
    lower_value = str(lower)
    if upper_value == NA_VALUE and lower_value == NA_VALUE:
        return RegionalResolution(NA_VALUE, "both_na")
    if lower_value == NA_VALUE:
        return RegionalResolution(upper_value, "upper_only")
    if upper_value == NA_VALUE:
        return RegionalResolution(lower_value, "lower_only")
    if upper_value == lower_value:
        return RegionalResolution(upper_value, "same")
    return RegionalResolution(NA_VALUE, "conflict")


def texture_value(row: Mapping[str, object], slot: GarmentSlot, attribute: str) -> str:
    """Return the canonical fabric/pattern value for ``slot`` from one raw row."""
    if attribute not in {"fabric", "pattern"}:
        raise ValueError(f"Unsupported texture attribute {attribute!r}.")

    direct = _DIRECT_COLUMNS.get(slot)
    if direct is not None:
        return str(row[direct[attribute]])

    if slot in ONE_PIECE_SLOTS:
        return resolve_regional_value(
            row[f"upper_{attribute}"], row[f"lower_{attribute}"]
        ).value

    return NA_VALUE
