"""Diagnostics for colour labels used by the SmolVLM ground-truth pipeline.

The regular dataset preflight proves structural consistency.  This module asks a
separate semantic question: do the coarse colour names produced from measured
CIELAB garment regions look plausible, or is the quantiser collapsing muted
chromatic colours onto achromatic names such as gray/black/white/beige?
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rmo.schemas import ColorName, OutfitDescription
from rmo.scoring.palette import reference_lab

__all__ = [
    "ColorAuditReport",
    "audit_color_descriptions",
    "chroma",
    "format_color_audit",
    "hue_degrees",
    "hue_family",
]

# These names are intended to be close to the neutral axis.  Brown and navy are
# stylistically treated as neutrals elsewhere in RMO, but they are deliberately
# excluded here because their LAB references are strongly chromatic.
_ACHROMATIC_NAMES: frozenset[ColorName] = frozenset(
    {ColorName.black, ColorName.white, ColorName.gray, ColorName.beige}
)

_CHROMATIC_NAMES: tuple[ColorName, ...] = (
    ColorName.red,
    ColorName.orange,
    ColorName.yellow,
    ColorName.chartreuse,
    ColorName.green,
    ColorName.spring_green,
    ColorName.cyan,
    ColorName.azure,
    ColorName.blue,
    ColorName.violet,
    ColorName.magenta,
    ColorName.rose,
)


def chroma(lab: tuple[float, float, float]) -> float:
    """Return CIE L*C*h chroma for one CIELAB sample."""
    return math.hypot(float(lab[1]), float(lab[2]))


def hue_degrees(lab: tuple[float, float, float]) -> float:
    """Return CIE L*C*h hue angle in degrees on [0, 360)."""
    return math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360.0


def _angular_distance(first: float, second: float) -> float:
    delta = abs(first - second) % 360.0
    return min(delta, 360.0 - delta)


def _reference_hue(name: ColorName) -> float:
    lab = reference_lab(name)
    if lab is None:  # pragma: no cover - guarded by the fixed name tuple above
        raise ValueError(f"No reference LAB value for {name.value!r}.")
    return hue_degrees(lab)


_REFERENCE_HUES: dict[ColorName, float] = {
    name: _reference_hue(name) for name in _CHROMATIC_NAMES
}


def hue_family(lab: tuple[float, float, float]) -> ColorName:
    """Return the chromatic reference whose *hue angle* is nearest to ``lab``.

    This is a diagnostic, not the production colour classifier.  Ignoring
    lightness and chroma makes it useful for exposing muted colours whose hue is
    clearly chromatic even when CIE76 distance selects an achromatic reference.
    """
    hue = hue_degrees(lab)
    return min(
        _CHROMATIC_NAMES,
        key=lambda name: (_angular_distance(hue, _REFERENCE_HUES[name]), name.value),
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[rank - 1])


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "median": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "max": round(max(values), 2),
    }


@dataclass(frozen=True, slots=True)
class SuspiciousColor:
    image_id: str
    garment_ref: str
    slot: str
    assigned: str
    hue_family: str
    lightness: float
    chroma: float
    hue_degrees: float
    lab: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ColorAuditReport:
    garments_with_lab: int
    assigned_names: dict[str, int]
    achromatic_assignments: int
    high_chroma_achromatic: int
    high_chroma_threshold: float
    chroma: dict[str, float]
    hue_families_high_chroma: dict[str, int]
    achromatic_by_hue_family: dict[str, int]
    per_assigned_name_chroma: dict[str, dict[str, float]]
    suspicious_examples: tuple[SuspiciousColor, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suspicious_examples"] = [asdict(item) for item in self.suspicious_examples]
        return payload


def audit_color_descriptions(
    descriptions: Iterable[OutfitDescription],
    *,
    high_chroma_threshold: float = 25.0,
    max_examples: int = 20,
) -> ColorAuditReport:
    """Audit measured garment colours without changing any training labels."""
    if high_chroma_threshold <= 0:
        raise ValueError("high_chroma_threshold must be positive.")
    if max_examples < 1:
        raise ValueError("max_examples must be positive.")

    assigned = Counter[str]()
    hue_families = Counter[str]()
    achromatic_hues = Counter[str]()
    per_name: dict[str, list[float]] = defaultdict(list)
    chromas: list[float] = []
    suspicious: list[SuspiciousColor] = []
    achromatic_count = 0
    high_chroma_achromatic = 0

    for description in descriptions:
        for garment in description.garments:
            if garment.color_lab is None:
                continue
            lab = tuple(float(value) for value in garment.color_lab)
            if not np.all(np.isfinite(np.asarray(lab, dtype=float))):
                continue
            value_chroma = chroma(lab)
            assigned[garment.color.value] += 1
            chromas.append(value_chroma)
            per_name[garment.color.value].append(value_chroma)

            if garment.color in _ACHROMATIC_NAMES:
                achromatic_count += 1

            if value_chroma >= high_chroma_threshold:
                family = hue_family(lab).value
                hue_families[family] += 1
                if garment.color in _ACHROMATIC_NAMES:
                    high_chroma_achromatic += 1
                    achromatic_hues[family] += 1
                    suspicious.append(
                        SuspiciousColor(
                            image_id=description.image_id,
                            garment_ref=garment.ref,
                            slot=garment.slot.value,
                            assigned=garment.color.value,
                            hue_family=family,
                            lightness=round(lab[0], 2),
                            chroma=round(value_chroma, 2),
                            hue_degrees=round(hue_degrees(lab), 2),
                            lab=(round(lab[0], 2), round(lab[1], 2), round(lab[2], 2)),
                        )
                    )

    suspicious.sort(key=lambda item: (-item.chroma, item.image_id, item.garment_ref))
    suspicious = suspicious[:max_examples]

    return ColorAuditReport(
        garments_with_lab=sum(assigned.values()),
        assigned_names=dict(sorted(assigned.items())),
        achromatic_assignments=achromatic_count,
        high_chroma_achromatic=high_chroma_achromatic,
        high_chroma_threshold=float(high_chroma_threshold),
        chroma=_distribution(chromas),
        hue_families_high_chroma=dict(sorted(hue_families.items())),
        achromatic_by_hue_family=dict(sorted(achromatic_hues.items())),
        per_assigned_name_chroma={
            name: _distribution(values) for name, values in sorted(per_name.items())
        },
        suspicious_examples=tuple(suspicious),
    )


def format_color_audit(report: ColorAuditReport) -> str:
    """Return a compact human-readable diagnostic report."""
    total = report.garments_with_lab
    achromatic_pct = 100.0 * report.achromatic_assignments / total if total else 0.0
    high_pct = 100.0 * report.high_chroma_achromatic / total if total else 0.0
    lines = [
        f"Garments with measured LAB: {total}",
        f"Assigned colour names: {report.assigned_names}",
        f"Achromatic-name assignments (black/white/gray/beige): "
        f"{report.achromatic_assignments} ({achromatic_pct:.2f}%)",
        f"Achromatic names with C* >= {report.high_chroma_threshold:g}: "
        f"{report.high_chroma_achromatic} ({high_pct:.2f}% of all measured garments)",
        f"Chroma C*: {report.chroma}",
        f"Hue families among C* >= {report.high_chroma_threshold:g}: "
        f"{report.hue_families_high_chroma}",
        "High-chroma samples currently assigned an achromatic name, by hue family: "
        f"{report.achromatic_by_hue_family}",
        "Per assigned-name chroma:",
    ]
    for name, stats in report.per_assigned_name_chroma.items():
        lines.append(f"  {name}: {stats}")
    lines.append("Most chromatic samples currently assigned black/white/gray/beige:")
    for item in report.suspicious_examples:
        lines.append(
            f"  {item.image_id} {item.garment_ref or item.slot}: assigned={item.assigned}, "
            f"hue_family={item.hue_family}, L*={item.lightness:.2f}, "
            f"C*={item.chroma:.2f}, h={item.hue_degrees:.2f}, LAB={item.lab}"
        )
    return "\n".join(lines)
