"""Cover the roast output contract and the generation interface."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rmo.roast.base import RoastGenerator
from rmo.schemas import (
    OutfitDescription,
    OutfitScore,
    Provenance,
    RoastOutput,
    SubScores,
    Tone,
)


def make_roast(**overrides: Any) -> RoastOutput:
    payload: dict[str, Any] = {
        "image_id": "fixture_000",
        "roast": "The jacket is having a very different evening from the shorts.",
        "suggestions": ["Swap the shorts for dark chinos."],
        "provenance": Provenance.fixture,
        "source_model": "fixture",
    }
    payload.update(overrides)
    return RoastOutput.model_validate(payload)


def test_roast_length_limits() -> None:
    assert len(make_roast(roast="a" * 1000).roast) == 1000
    with pytest.raises(ValidationError):
        make_roast(roast="a" * 1001)
    with pytest.raises(ValidationError):
        make_roast(roast="")


def test_suggestions_count_limits() -> None:
    assert len(make_roast(suggestions=["one"]).suggestions) == 1
    assert len(make_roast(suggestions=[f"idea {n}" for n in range(5)]).suggestions) == 5
    with pytest.raises(ValidationError):
        make_roast(suggestions=[])
    with pytest.raises(ValidationError):
        make_roast(suggestions=[f"idea {n}" for n in range(6)])


def test_each_suggestion_carries_text() -> None:
    with pytest.raises(ValidationError):
        make_roast(suggestions=[""])
    with pytest.raises(ValidationError):
        make_roast(suggestions=["   "])
    with pytest.raises(ValidationError):
        make_roast(suggestions=["Swap the shorts.", ""])


def test_suggestion_length_limit() -> None:
    assert len(make_roast(suggestions=["a" * 280]).suggestions[0]) == 280
    with pytest.raises(ValidationError):
        make_roast(suggestions=["a" * 281])


def test_blank_list_entries_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_roast(grounded_garments=[""])
    with pytest.raises(ValidationError):
        make_roast(safety_flags=["   "])


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_roast(punchline="none")


def test_tone_defaults_to_playful() -> None:
    assert make_roast().tone is Tone.playful


def test_safety_flags_default_to_empty() -> None:
    assert make_roast().safety_flags == []
    assert make_roast().grounded_garments == []


def test_compliment_is_a_tone() -> None:
    assert Tone.compliment.value == "compliment"


def test_provenance_must_be_stated() -> None:
    with pytest.raises(ValidationError):
        RoastOutput(
            image_id="fixture_000",
            roast="The jacket and the shorts have not spoken in years.",
            suggestions=["Swap the shorts for dark chinos."],
            source_model="fixture",
        )


def test_unlisted_tone_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_roast(tone="withering")


class ConstantRoast(RoastGenerator):
    name = "constant"

    def generate(self, description: OutfitDescription, score: OutfitScore) -> RoastOutput:
        return make_roast(image_id=description.image_id, tone=Tone.gentle)


def test_roast_generator_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RoastGenerator()


def test_subclass_without_generate_cannot_be_instantiated() -> None:
    class Incomplete(RoastGenerator):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_generate_takes_both_the_description_and_the_score() -> None:
    description = OutfitDescription.model_validate(
        {
            "image_id": "fixture_002",
            "garments": [{"slot": "upper", "category": "graphic tee"}],
            "source_model": "fixture",
        }
    )
    score = OutfitScore.model_validate(
        {
            "image_id": "fixture_002",
            "overall": 41.0,
            "subscores": SubScores(
                color_harmony=40.0,
                formality_consistency=30.0,
                seasonality=55.0,
                proportion=39.0,
            ),
            "provenance": Provenance.fixture,
            "source_model": "fixture",
        }
    )
    generated = ConstantRoast().generate(description, score)
    assert generated.image_id == "fixture_002"
    assert generated.tone is Tone.gentle
