"""Deterministic text format exchanged with the fine-tuned perception VLM.

The format stays deliberately small and line-oriented so a compact VLM can learn
it reliably while every schema attribute remains explicitly named.  Free-form
zero-shot output is still accepted by :mod:`rmo.perception.postprocess` as a
fallback; this module only serialises the canonical keyed form used for training.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from rmo.schemas import Garment, OutfitDescription

__all__ = ["serialize_garment", "serialize_description"]

_FIELD_ORDER: tuple[str, ...] = (
    "color",
    "pattern",
    "fabric",
    "sleeve_length",
    "length",
    "neckline",
)


def _value(value: object) -> str:
    """Return a schema enum/string as the text emitted to the VLM target."""
    return str(getattr(value, "value", value))


def _safe_category(category: str) -> str:
    """Keep one garment on one record without inventing an escaping grammar."""
    return " ".join(category.replace("|", "/").split())


def serialize_garment(
    garment: Garment,
    *,
    omit: Collection[str] = (),
) -> str:
    """Return one unambiguous keyed record for ``garment``.

    ``omit`` exists for attributes that are intentionally unsupervised for one
    training example (for example a conflicting dress upper/lower fabric label).
    Inapplicable shape fields are omitted automatically because the schema stores
    them as ``None``.
    """
    omitted = frozenset(omit)
    fields = [_safe_category(garment.category)]
    for name in _FIELD_ORDER:
        if name in omitted:
            continue
        value = getattr(garment, name)
        if value is None:
            continue
        fields.append(f"{name}={_value(value)}")
    return f"{garment.slot.value}: " + " | ".join(fields)


def serialize_description(
    description: OutfitDescription,
    *,
    omitted_fields: Mapping[str, Collection[str]] | None = None,
) -> str:
    """Return all garments as deterministic one-line keyed records."""
    omissions = omitted_fields or {}
    return "\n".join(
        serialize_garment(garment, omit=omissions.get(garment.ref, ()))
        for garment in description.garments
    )
