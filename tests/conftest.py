"""Shared access to the committed fixture corpus."""

from __future__ import annotations

import pytest

from rmo import fixtures
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

EXPECTED_IDS = [
    *(f"fixture_{index:03d}" for index in range(17)),
    *(f"fx_adv_{index:02d}" for index in range(3)),
    *(f"fx_deg_{index:02d}" for index in range(8)),
]

SAMPLE_ID = "fixture_000"


@pytest.fixture(scope="module")
def descriptions() -> dict[str, OutfitDescription]:
    """Return the committed descriptions."""
    return fixtures.load_records("outfit_descriptions.jsonl", OutfitDescription)


@pytest.fixture(scope="module")
def scores() -> dict[str, OutfitScore]:
    """Return the committed scores."""
    return fixtures.load_records("outfit_scores.jsonl", OutfitScore)


@pytest.fixture(scope="module")
def roasts() -> dict[str, RoastOutput]:
    """Return the committed roasts."""
    return fixtures.load_records("roast_outputs.jsonl", RoastOutput)
