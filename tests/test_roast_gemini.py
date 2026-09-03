"""Cover Gemini roast generation without making network calls."""

from __future__ import annotations

from types import SimpleNamespace

from rmo.roast.gemini import GeminiRoaster
from rmo.schemas import OutfitDescription, OutfitScore


class FakeInteractions:
    def __init__(self, *, output_text: str | None = None, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, interactions: FakeInteractions) -> None:
        self.interactions = interactions


def test_gemini_roaster_returns_generated_copy(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(
        output_text=(
            '{"roast":"Your palette has more plot twists than the outfit can support.",'
            '"suggestions":["Keep one accent colour and neutralise the rest."]}'
        )
    )
    roaster = GeminiRoaster(client=FakeClient(interactions), model="gemini-3.6-flash")

    image_id = "fixture_002"
    roast = roaster.generate(descriptions[image_id], scores[image_id])

    assert roast.source_model == GeminiRoaster.name
    assert roast.image_id == image_id
    assert roast.provenance == descriptions[image_id].provenance
    assert roast.safety_flags == []
    assert set(roast.grounded_garments) <= set(descriptions[image_id].refs())
    assert interactions.calls[0]["model"] == "gemini-3.6-flash"
    response_format = interactions.calls[0]["response_format"]
    assert response_format["mime_type"] == "application/json"


def test_gemini_roaster_falls_back_on_api_error(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(error=RuntimeError("network down"))
    roaster = GeminiRoaster(client=FakeClient(interactions))

    roast = roaster.generate(descriptions["fixture_002"], scores["fixture_002"])
    assert roast.source_model == "rule_roaster"


def test_gemini_roaster_falls_back_on_invalid_structured_output(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(output_text='{"roast":"Missing suggestions"}')
    roaster = GeminiRoaster(client=FakeClient(interactions))

    roast = roaster.generate(descriptions["fixture_002"], scores["fixture_002"])
    assert roast.source_model == "rule_roaster"


def test_gemini_roaster_falls_back_on_unsafe_copy(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(
        output_text=(
            '{"roast":"You would look better if you lost some weight.",'
            '"suggestions":["Keep the blouse."]}'
        )
    )
    roaster = GeminiRoaster(client=FakeClient(interactions))

    roast = roaster.generate(descriptions["fixture_002"], scores["fixture_002"])
    assert roast.source_model == "rule_roaster"


def test_unscorable_record_does_not_call_gemini(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(
        output_text='{"roast":"unused","suggestions":["unused"]}'
    )
    roaster = GeminiRoaster(client=FakeClient(interactions))

    roast = roaster.generate(descriptions["fx_deg_05"], scores["fx_deg_05"])
    assert roast.source_model == "rule_roaster"
    assert interactions.calls == []
