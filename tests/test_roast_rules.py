"""Tests for the rule-based roast integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rmo.roast.rules import RuleBasedRoaster
from rmo.roast.safety import flag_text
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput, Tone


def test_rule_roaster_returns_a_schema_valid_output(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    image_id = "fixture_002"
    roast = RuleBasedRoaster().generate(descriptions[image_id], scores[image_id])
    assert RoastOutput.model_validate(roast.model_dump(mode="json")) == roast
    assert roast.image_id == image_id
    assert roast.provenance == descriptions[image_id].provenance
    assert roast.source_model == RuleBasedRoaster.name
    assert set(roast.grounded_garments) <= set(descriptions[image_id].refs())
    assert roast.suggestions


def test_rule_roaster_uses_compliment_tone_for_a_clean_high_scoring_outfit(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    image_id = "fx_adv_02"
    roast = RuleBasedRoaster().generate(descriptions[image_id], scores[image_id])
    assert roast.tone is Tone.compliment
    assert roast.suggestions == ["Keep it exactly as it is."]


def test_rule_roaster_handles_an_unscorable_record_gently(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    image_id = "fx_deg_05"
    roast = RuleBasedRoaster().generate(descriptions[image_id], scores[image_id])
    assert roast.tone is Tone.gentle
    assert roast.grounded_garments == []
    assert "Re-run perception" in roast.suggestions[0]


@pytest.mark.parametrize("image_id", ["fixture_000", "fixture_003", "fixture_009", "fx_adv_00"])
def test_rule_roaster_grounded_refs_are_subset_of_the_description(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    roast = RuleBasedRoaster().generate(descriptions[image_id], scores[image_id])
    assert set(roast.grounded_garments) <= set(descriptions[image_id].refs())


def test_safety_flagger_matches_the_probe_corpus() -> None:
    path = Path("data/fixtures/safety_probes.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        probe = json.loads(line)
        assert bool(flag_text(probe["text"])) is probe["must_flag"], probe["probe_id"]