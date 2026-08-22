"""Cover the outfit description contract and the perception interface."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rmo.imaging import ImageInput
from rmo.perception.base import PerceptionModel
from rmo.schemas import Garment, GarmentSlot, OutfitDescription, Provenance


def make_description(**overrides: Any) -> OutfitDescription:
    payload: dict[str, Any] = {
        "image_id": "fixture_000",
        "garments": [{"slot": "upper", "category": "graphic tee"}],
        "source_model": "fixture",
    }
    payload.update(overrides)
    return OutfitDescription.model_validate(payload)


def test_blank_refs_are_numbered_per_slot() -> None:
    description = make_description(
        garments=[
            {"slot": "upper", "category": "graphic tee"},
            {"slot": "outer", "category": "biker jacket"},
            {"slot": "lower", "category": "chinos"},
        ]
    )
    assert description.refs() == ["upper_0", "outer_0", "lower_0"]


def test_repeated_slot_numbers_from_zero() -> None:
    description = make_description(
        garments=[{"slot": "jewelry", "category": f"ring {n}"} for n in range(4)]
    )
    assert description.refs() == ["jewelry_0", "jewelry_1", "jewelry_2", "jewelry_3"]


def test_duplicate_explicit_refs_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_description(
            garments=[
                {"ref": "upper_0", "slot": "upper", "category": "graphic tee"},
                {"ref": "upper_0", "slot": "upper", "category": "oxford shirt"},
            ]
        )


def test_outfit_needs_at_least_one_garment() -> None:
    with pytest.raises(ValidationError):
        make_description(garments=[])


def test_garment_numeric_fields_are_bounded() -> None:
    with pytest.raises(ValidationError):
        Garment(slot=GarmentSlot.upper, category="graphic tee", confidence=1.1)
    with pytest.raises(ValidationError):
        Garment(slot=GarmentSlot.upper, category="graphic tee", area_fraction=-0.1)


def test_caption_length_limit() -> None:
    assert len(make_description(caption="a" * 2000).caption) == 2000
    with pytest.raises(ValidationError):
        make_description(caption="a" * 2001)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_description(mood="chaotic")


def test_by_slot_keeps_only_its_slot_in_order() -> None:
    description = make_description(
        garments=[
            {"slot": "upper", "category": "graphic tee"},
            {"slot": "lower", "category": "chinos"},
            {"slot": "upper", "category": "oxford shirt"},
        ]
    )
    upper = description.by_slot(GarmentSlot.upper)
    assert [garment.category for garment in upper] == ["graphic tee", "oxford shirt"]
    assert description.refs() == ["upper_0", "lower_0", "upper_1"]


def test_enum_members_serialise_to_plain_strings() -> None:
    assert Provenance.gt.value == "gt"


def test_color_lab_source_defaults_to_none() -> None:
    assert Garment(slot=GarmentSlot.upper, category="graphic tee").color_lab_source is None


def test_color_lab_source_rejects_unlisted_value() -> None:
    with pytest.raises(ValidationError):
        Garment(slot=GarmentSlot.upper, category="graphic tee", color_lab_source="guessed")


class ConstantPerception(PerceptionModel):
    name = "constant"

    def predict(self, image: ImageInput) -> OutfitDescription:
        return make_description(image_id=str(image))


def test_perception_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        PerceptionModel()


def test_subclass_without_predict_cannot_be_instantiated() -> None:
    class Incomplete(PerceptionModel):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_predict_batch_returns_one_description_per_image_in_order() -> None:
    images = ["a.png", "b.png", "c.png"]
    described = ConstantPerception().predict_batch(images)
    assert [description.image_id for description in described] == images
