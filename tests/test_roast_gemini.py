"""Cover Gemini roast generation without making network calls."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from rmo.roast.gemini import GeminiRoaster
from rmo.roast.rules import RuleBasedRoaster
from rmo.schemas import OutfitDescription, OutfitScore

VALID_OUTPUT = '{"roast":"Bold choices all round.","suggestions":["Pick one accent colour."]}'


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
    assert roast.safety_flags == ["safety:body"]


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


def test_missing_credentials_fall_back_to_the_rule_roaster(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(GeminiRoaster, "_load_dotenv", staticmethod(lambda: None))

    with caplog.at_level(logging.INFO, logger="rmo.roast.gemini"):
        roaster = GeminiRoaster()

    assert ("rmo.roast.gemini", "WARNING") in [
        (record.name, record.levelname) for record in caplog.records
    ]
    roast = roaster.generate(descriptions["fixture_002"], scores["fixture_002"])
    assert roast.source_model == "rule_roaster"


def test_the_prompt_is_rendered_from_the_config_file(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    tmp_path,
) -> None:
    config_path = tmp_path / "roast.yaml"
    config_path.write_text(
        "persona: |-\n"
        "  PERSONA MARKER\n"
        "prompt_template: |-\n"
        "  tone={tone}\n"
        "  caption={caption}\n"
        "  garments={garments}\n"
        "  issues={issues}\n",
        encoding="utf-8",
    )
    interactions = FakeInteractions(output_text=VALID_OUTPUT)
    roaster = GeminiRoaster(client=FakeClient(interactions), config_path=config_path)

    image_id = "fixture_002"
    description = descriptions[image_id]
    fallback = RuleBasedRoaster().generate(description, scores[image_id])
    roaster.generate(description, scores[image_id])

    prompt = interactions.calls[0]["input"]
    assert prompt.startswith("PERSONA MARKER\n\n")
    assert f"tone={fallback.tone.value}" in prompt
    assert f"caption={description.caption}" in prompt
    assert f"garments=- {description.garments[0].ref}:" in prompt
    assert "issues=- " in prompt


def test_the_prompt_uses_the_tone_value(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(output_text=VALID_OUTPUT)
    roaster = GeminiRoaster(client=FakeClient(interactions))

    image_id = "fixture_002"
    fallback = RuleBasedRoaster().generate(descriptions[image_id], scores[image_id])
    roaster.generate(descriptions[image_id], scores[image_id])

    prompt = interactions.calls[0]["input"]
    assert "Tone." not in prompt
    assert f"Requested tone: {fallback.tone.value}" in prompt


def test_the_request_carries_no_image_data(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
) -> None:
    interactions = FakeInteractions(output_text=VALID_OUTPUT)
    roaster = GeminiRoaster(client=FakeClient(interactions))

    image_id = "fixture_002"
    description = descriptions[image_id]
    roaster.generate(description, scores[image_id])

    call = interactions.calls[0]
    assert set(call) == {"model", "input", "response_format"}
    assert not any(isinstance(value, (bytes, bytearray)) for value in call.values())
    assert description.image_path
    assert description.image_path not in call["input"]


def test_the_request_and_response_are_logged(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    caplog,
) -> None:
    interactions = FakeInteractions(output_text=VALID_OUTPUT)
    roaster = GeminiRoaster(client=FakeClient(interactions))

    caplog.set_level(logging.DEBUG, logger="rmo.roast.gemini")
    roaster.generate(descriptions["fixture_002"], scores["fixture_002"])

    prompt = interactions.calls[0]["input"]
    messages = [record.getMessage() for record in caplog.records]
    assert any(prompt in message for message in messages)
    assert any(VALID_OUTPUT in message for message in messages)


def test_the_api_key_is_never_logged(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sentinel-key-value")
    interactions = FakeInteractions(output_text=VALID_OUTPUT)
    roaster = GeminiRoaster(client=FakeClient(interactions))

    caplog.set_level(logging.DEBUG, logger="rmo.roast.gemini")
    roaster.generate(descriptions["fixture_002"], scores["fixture_002"])

    assert all(
        "sentinel-key-value" not in record.getMessage() for record in caplog.records
    )
