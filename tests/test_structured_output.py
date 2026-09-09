"""The fine-tuning text contract must round-trip without attribute ambiguity."""

from __future__ import annotations

import itertools

from rmo.perception.postprocess import parse_description
from rmo.perception.structured_output import serialize_description, serialize_garment
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


def _parse(text: str) -> OutfitDescription:
    return parse_description(text, image_id="roundtrip", source_model="fixture")


def test_keyed_fields_disambiguate_overlapping_enum_values() -> None:
    text = (
        "upper: top | color=blue | pattern=floral | fabric=other | "
        "sleeve_length=na | neckline=na"
    )
    garment = _parse(text).garments[0]
    assert garment.pattern is Pattern.floral
    assert garment.fabric is Fabric.other
    assert garment.sleeve_length is SleeveLength.na
    assert garment.neckline is Neckline.na


def test_pattern_other_does_not_become_fabric_other() -> None:
    garment = _parse(
        "upper: top | color=blue | pattern=other | fabric=na | "
        "sleeve_length=short | neckline=round"
    ).garments[0]
    assert garment.pattern is Pattern.other
    assert garment.fabric is Fabric.na


def test_all_fabric_pattern_pairs_round_trip() -> None:
    for fabric, pattern in itertools.product(Fabric, Pattern):
        original = OutfitDescription(
            image_id="roundtrip",
            source_model="gt",
            garments=[
                Garment(
                    slot=GarmentSlot.upper,
                    category="top",
                    color=ColorName.blue,
                    fabric=fabric,
                    pattern=pattern,
                    sleeve_length=SleeveLength.short,
                    neckline=Neckline.round,
                )
            ],
        )
        parsed = _parse(serialize_description(original)).garments[0]
        assert parsed.fabric is fabric
        assert parsed.pattern is pattern


def test_dress_round_trip_preserves_all_applicable_shape_fields() -> None:
    garment = Garment(
        slot=GarmentSlot.dress,
        category="dress",
        color=ColorName.red,
        pattern=Pattern.floral,
        fabric=Fabric.chiffon,
        sleeve_length=SleeveLength.long,
        length=LowerLength.three_quarter,
        neckline=Neckline.v_shape,
    )
    parsed = _parse(serialize_garment(garment)).garments[0]
    assert parsed.color is ColorName.red
    assert parsed.pattern is Pattern.floral
    assert parsed.fabric is Fabric.chiffon
    assert parsed.sleeve_length is SleeveLength.long
    assert parsed.length is LowerLength.three_quarter
    assert parsed.neckline is Neckline.v_shape


def test_inapplicable_shape_fields_are_not_serialized() -> None:
    line = serialize_garment(
        Garment(
            slot=GarmentSlot.footwear,
            category="footwear",
            color=ColorName.black,
            pattern=Pattern.na,
            fabric=Fabric.na,
        )
    )
    assert "sleeve_length=" not in line
    assert "length=" not in line
    assert "neckline=" not in line


def test_intentionally_unsupervised_field_can_be_omitted() -> None:
    garment = Garment(
        ref="dress_0",
        slot=GarmentSlot.dress,
        category="dress",
        color=ColorName.red,
        pattern=Pattern.na,
        fabric=Fabric.na,
        sleeve_length=SleeveLength.long,
        length=LowerLength.long,
        neckline=Neckline.round,
    )
    description = OutfitDescription(
        image_id="roundtrip", source_model="gt", garments=[garment]
    )
    text = serialize_description(
        description,
        omitted_fields={"dress_0": {"pattern", "fabric"}},
    )
    assert "pattern=" not in text
    assert "fabric=" not in text
    parsed = _parse(text).garments[0]
    assert parsed.pattern is Pattern.na
    assert parsed.fabric is Fabric.na


def test_legacy_free_form_parser_remains_supported() -> None:
    garment = _parse("upper: tee, white, graphic, cotton, short sleeves").garments[0]
    assert garment.category == "tee"
    assert garment.color is ColorName.white
    assert garment.pattern is Pattern.graphic
    assert garment.fabric is Fabric.cotton
    assert garment.sleeve_length is SleeveLength.short
