"""Turn raw VLM text into a schema-valid outfit description.

Never raises on model output; unparseable text yields a fallback description.
"""

from __future__ import annotations

import logging
import re
from difflib import get_close_matches
from enum import Enum
from functools import lru_cache
from pathlib import Path

from rmo.config import load_perception_config
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

log = logging.getLogger(__name__)

__all__ = ["parse_description"]

_Record = tuple[GarmentSlot, str, str | None]

_FUZZY_CUTOFF = 0.85
_MIN_FUZZY_LENGTH = 4
_MAX_FUZZY_WORDS = 2
_MAX_CATEGORY = 64
_MAX_CAPTION = 2000

_ATTRIBUTES: tuple[tuple[str, type[Enum]], ...] = (
    ("color", ColorName),
    ("pattern", Pattern),
    ("fabric", Fabric),
    ("sleeve_length", SleeveLength),
    ("length", LowerLength),
    ("neckline", Neckline),
)

_UNIVERSAL_ATTRIBUTES: tuple[str, ...] = ("color", "pattern", "fabric")

_SHAPE_ATTRIBUTES: dict[GarmentSlot, tuple[str, ...]] = {
    GarmentSlot.upper: ("sleeve_length", "neckline"),
    GarmentSlot.outer: ("sleeve_length", "neckline"),
    GarmentSlot.dress: ("sleeve_length", "neckline", "length"),
    GarmentSlot.romper: ("sleeve_length", "neckline", "length"),
    GarmentSlot.lower: ("length",),
}

_NOISE = re.compile(r"[*`#]+")
_BULLET = re.compile(r"^\s*(?:[-\u2013\u2014\u2022>]\s*|\d+[.)]\s*)+")
_KEY = re.compile(r"([A-Za-z][A-Za-z \-_/]*?)\s*[:=]\s*")
_FIELD = re.compile(r"[,|]")
_LINE = re.compile(r"[\r\n;]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_CLAUSE = re.compile(r"[,.;:!?\r\n]|\band\b|\bwith\b|\bover\b|\bunder\b|\bplus\b")

_PHRASE_WORDS = 3
_PHRASE_STOPWORDS = frozenset(
    {
        "a", "an", "the", "this", "that", "her", "his", "its", "their",
        "he", "she", "they", "person", "woman", "man", "girl", "boy", "model",
        "is", "are", "was", "were", "has", "have", "in", "on", "of", "s",
        "wearing", "wears", "worn", "dressed", "holds", "holding",
        "carries", "carrying", "pair", "some",
    }
)


def _normalise(text: str) -> str:
    """Return ``text`` lowercased with every run of non-alphanumerics collapsed to one space."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@lru_cache(maxsize=None)
def _lookups(config_path: Path | None) -> dict[str, dict[str, Enum]]:
    """Return the normalised value and synonym tables for each attribute, cached per config."""
    tables: dict[str, dict[str, Enum]] = {
        name: {_normalise(member.value): member for member in enum_type}
        for name, enum_type in (("slot", GarmentSlot), *_ATTRIBUTES)
    }
    synonyms = load_perception_config(config_path).get("synonyms", {})
    for name, table in tables.items():
        for raw_key, raw_value in synonyms.get(name, {}).items():
            target = table.get(_normalise(str(raw_value)))
            if target is not None:
                table[_normalise(str(raw_key))] = target
    return tables


def _resolve(candidate: str, table: dict[str, Enum]) -> Enum | None:
    """Return the vocabulary member ``candidate`` names, exactly or by close match."""
    exact = table.get(candidate)
    if exact is not None:
        return exact
    if len(candidate) < _MIN_FUZZY_LENGTH or candidate.count(" ") >= _MAX_FUZZY_WORDS:
        return None
    matches = get_close_matches(candidate, table, n=1, cutoff=_FUZZY_CUTOFF)
    return table[matches[0]] if matches else None


def _candidates(field: str) -> list[tuple[range, str]]:
    """Return the word spans of a normalised field and their text, widest span first."""
    words = field.split()
    if not words:
        return []
    spans = dict.fromkeys(
        (start, start + width)
        for width in (len(words), 3, 2, 1)
        for start in range(len(words) - width + 1)
    )
    return [(range(start, stop), " ".join(words[start:stop])) for start, stop in spans]


def _slot_records(text: str, slots: dict[str, Enum]) -> list[_Record]:
    """Return ``(slot, body, category)`` for every line whose leading key names a garment slot."""
    records: list[_Record] = []
    for fragment in _LINE.split(text):
        line = _BULLET.sub("", _NOISE.sub("", fragment)).strip()
        if not line:
            continue
        for match in _KEY.finditer(line):
            slot = _resolve(_normalise(match.group(1)), slots)
            if isinstance(slot, GarmentSlot):
                records.append((slot, line[match.end() :], None))
                break
    return records


def _prose_records(text: str, slots: dict[str, Enum]) -> list[_Record]:
    """Return ``(slot, clause, phrase)`` for prose clauses ending in a known garment word."""
    records: list[_Record] = []
    for clause in _CLAUSE.split(text):
        words = _normalise(clause).split()
        # a clause of nothing but slot words is an echoed vocabulary list, not a description
        if all(word in slots for word in words):
            continue
        for index in reversed(range(len(words))):
            slot = slots.get(words[index])
            if not isinstance(slot, GarmentSlot) or slot is GarmentSlot.other:
                continue
            head = words[max(0, index - _PHRASE_WORDS + 1) : index + 1]
            kept = [word for word in head if word not in _PHRASE_STOPWORDS]
            records.append((slot, clause, " ".join(kept) or words[index]))
            break
    return records


def _category(
    slot: GarmentSlot,
    fields: list[str],
    tables: dict[str, dict[str, Enum]],
    applicable: tuple[str, ...],
) -> str:
    """Return the garment category named by the first field, else the slot name."""
    if not fields:
        return slot.value
    head = _normalise(fields[0])
    if any(head in tables[name] for name in applicable):
        return slot.value
    category = fields[0][:_MAX_CATEGORY]
    return category if _normalise(category) else slot.value


def _garment(
    slot: GarmentSlot,
    body: str,
    tables: dict[str, dict[str, Enum]],
    category: str | None = None,
) -> Garment:
    """Return the garment described by one record body."""
    shape = _SHAPE_ATTRIBUTES.get(slot, ())
    applicable = _UNIVERSAL_ATTRIBUTES + shape
    fields = [field for field in (part.strip(" .\t") for part in _FIELD.split(body)) if field]

    values: dict[str, Enum] = {}
    for field in fields:
        taken: set[int] = set()
        for span, candidate in _candidates(_normalise(field)):
            if taken.intersection(span):
                continue
            for name, _ in _ATTRIBUTES:
                if name not in applicable or name in values:
                    continue
                resolved = _resolve(candidate, tables[name])
                if resolved is not None:
                    values[name] = resolved
                    taken.update(span)
                    break

    for name, enum_type in _ATTRIBUTES:
        if name in shape and name not in values:
            values[name] = enum_type("na")

    named = category[:_MAX_CATEGORY] if category else _category(slot, fields, tables, applicable)
    return Garment(slot=slot, category=named, **values)


def parse_description(
    text: str,
    *,
    image_id: str,
    source_model: str,
    image_path: str = "",
    config_path: Path | None = None,
) -> OutfitDescription:
    """Return the outfit described by ``text``, falling back when nothing parses."""
    # lone surrogates survive tokenizer decoding but pydantic rejects them as strings
    text = text.encode("utf-8", "replace").decode("utf-8")
    tables = _lookups(config_path)

    records = _slot_records(text, tables["slot"]) or _prose_records(text, tables["slot"])

    garments: list[Garment] = []
    seen: set[tuple[GarmentSlot, str]] = set()
    for slot, body, category in records:
        garment = _garment(slot, body, tables, category)
        key = (slot, _normalise(garment.category))
        if key in seen:
            continue
        seen.add(key)
        garments.append(garment)

    if not garments:
        log.debug("no garments parsed from %d characters", len(text))
        garments = [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)]

    return OutfitDescription(
        image_id=image_id,
        image_path=image_path,
        garments=garments,
        caption=text.strip()[:_MAX_CAPTION],
        source_model=source_model,
    )
