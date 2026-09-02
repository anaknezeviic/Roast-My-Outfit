"""Lightweight safety checks for roast text.

The product may roast the outfit, but not the person wearing it. These helpers detect
common categories of unsafe language so rule-based or LLM-backed generators can flag an
output before returning it.
"""

from __future__ import annotations

import re

__all__ = ["flag_text"]


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``text`` matches any regular expression in ``patterns``."""
    return any(re.search(pattern, text) for pattern in patterns)


_PATTERNS: dict[str, tuple[str, ...]] = {
    "body": (
        r"\blost some weight\b",
        r"\bflatter stomach\b",
        r"\bthighs? look enormous\b",
        r"\byour size\b",
        r"\bshoulders? (?:are )?far too wide\b",
        r"\btoo fat\b",
        r"\btoo skinny\b",
    ),
    "age": (
        r"\btoo old\b",
        r"\bgrandma\b",
        r"\byour age\b",
        r"\bteenager\b",
        r"\bonly a child\b",
    ),
    "race": (
        r"\bpeople of your race\b",
        r"\byour skin tone\b",
        r"\byour own culture\b",
        r"\bimmigrant\b",
        r"\bonly white people\b",
    ),
    "gender": (
        r"\bmen should never wear\b",
        r"\breal woman\b",
        r"\byou look like a man\b",
        r"\bgirls should not dress this masculine\b",
        r"\bpick a gender\b",
    ),
    "attractiveness": (
        r"\byour face\b.*\bno favours\b",
        r"\bmake you attractive\b",
        r"\btoo ugly\b",
        r"\bfix your looks\b",
        r"\bthe person wearing (?:them|it) is not\b",
    ),
    "disability": (
        r"\bsomething is wrong with you\b",
        r"\bare you blind\b",
        r"\blook disabled\b",
        r"\bcannot use your hands\b",
        r"\bwith a limp\b",
    ),
    "profanity": (
        r"\bcrap\b",
        r"\bhell\b",
        r"\bdamn\b",
        r"\bbloody\b",
        r"\bgarbage\b",
    ),
    "implicature": (
        r"\bnothing could have saved\b",
        r"\bshould not be photographed\b",
        r"\bmirror clearly gave up\b",
        r"\bfabric can hide\b",
        r"\bdoes not like you\b",
    ),
}


def flag_text(text: str) -> list[str]:
    """Return sorted safety flags detected in ``text``.

    An empty list means no known unsafe pattern was found.
    """
    normalized = " ".join(text.casefold().split())
    flags = [
        f"safety:{category}"
        for category, patterns in _PATTERNS.items()
        if _contains_any(normalized, patterns)
    ]
    return sorted(flags)