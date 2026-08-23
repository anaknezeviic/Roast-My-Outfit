"""Cover parsing raw VLM output into an outfit description."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rmo.perception import postprocess
from rmo.perception.postprocess import parse_description
from rmo.schemas import (
    SCHEMA_VERSION,
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
    Provenance,
    SleeveLength,
)

CLEAN = (
    "upper: t-shirt, white, graphic, cotton, short sleeves\n"
    "lower: jeans, blue, pure color, denim, long\n"
    "footwear: sneakers, black, pure color, leather, na"
)


def parse(text: str) -> OutfitDescription:
    return parse_description(text, image_id="fixture_000", source_model="fixture")


def max_length(model: type, field: str) -> int:
    return next(
        rule.max_length
        for rule in model.model_fields[field].metadata
        if hasattr(rule, "max_length")
    )


def test_clean_output_yields_one_garment_per_line() -> None:
    description = parse(CLEAN)
    assert [garment.slot for garment in description.garments] == [
        GarmentSlot.upper,
        GarmentSlot.lower,
        GarmentSlot.footwear,
    ]
    assert description.refs() == ["upper_0", "lower_0", "footwear_0"]


def test_clean_output_populates_the_upper_garment() -> None:
    garment = parse(CLEAN).garments[0]
    assert garment.category == "t-shirt"
    assert garment.color is ColorName.white
    assert garment.pattern is Pattern.graphic
    assert garment.fabric is Fabric.cotton
    assert garment.sleeve_length is SleeveLength.short


def test_clean_output_populates_the_lower_garment() -> None:
    garment = parse(CLEAN).garments[1]
    assert garment.category == "jeans"
    assert garment.color is ColorName.blue
    assert garment.pattern is Pattern.pure_color
    assert garment.fabric is Fabric.denim
    assert garment.length is LowerLength.long


def test_parsed_garments_keep_full_confidence() -> None:
    assert all(garment.confidence == 1.0 for garment in parse(CLEAN).garments)


def test_description_metadata_is_carried_through() -> None:
    description = parse(CLEAN)
    assert description.image_id == "fixture_000"
    assert description.source_model == "fixture"
    assert description.provenance is Provenance.predicted
    assert description.schema_version == SCHEMA_VERSION


def test_caption_carries_the_raw_generation() -> None:
    assert parse(CLEAN).caption == CLEAN


def test_caption_is_truncated_to_the_schema_limit() -> None:
    assert len(parse("upper: " + "x" * 5000).caption) == 2000


def test_synonyms_map_onto_the_vocabulary() -> None:
    garment = parse("top: tee, navy blue, plain, denim, v-neck").garments[0]
    assert garment.slot is GarmentSlot.upper
    assert garment.color is ColorName.navy
    assert garment.pattern is Pattern.pure_color
    assert garment.fabric is Fabric.denim
    assert garment.neckline is Neckline.v_shape


def test_underscored_enum_spelling_parses() -> None:
    garment = parse("upper: tee, white, pure_color, cotton, na").garments[0]
    assert garment.pattern is Pattern.pure_color


def test_fuzzy_matching_absorbs_plurals() -> None:
    garment = parse("upper: shirt, stripes, cottons, na").garments[0]
    assert garment.pattern is Pattern.striped
    assert garment.fabric is Fabric.cotton


def test_fuzzy_matching_absorbs_a_single_typo() -> None:
    garment = parse("upper: tee, blue, striped, cotton, sleaveless").garments[0]
    assert garment.sleeve_length is SleeveLength.sleeveless


def test_item_names_do_not_hijack_an_attribute() -> None:
    garment = parse("upper: blouse, white, graphic, cotton, na").garments[0]
    assert garment.category == "blouse"
    assert garment.color is ColorName.white


def test_item_names_do_not_invent_a_shape_attribute() -> None:
    garment = parse("upper: t-shirt, white, graphic, cotton, na").garments[0]
    assert garment.sleeve_length is SleeveLength.na


def test_one_field_fills_at_most_one_attribute() -> None:
    garment = parse("dress: gown, red, floral, chiffon, long sleeves, na").garments[0]
    assert garment.sleeve_length is SleeveLength.long
    assert garment.length is LowerLength.na


def test_short_tokens_are_not_fuzzy_matched() -> None:
    garment = parse("upper: tee, ab, cd, ef, gh").garments[0]
    assert garment.color is ColorName.unknown
    assert garment.pattern is Pattern.na
    assert garment.fabric is Fabric.na


def test_multi_word_attributes_win_over_their_words() -> None:
    garment = parse("upper: gown, spring green, color block, chiffon, na").garments[0]
    assert garment.color is ColorName.spring_green
    assert garment.pattern is Pattern.color_block


def test_three_word_attributes_are_found_inside_a_longer_field() -> None:
    garment = parse("upper: tee, white, graphic, cotton, with three quarter sleeves").garments[0]
    assert garment.sleeve_length is SleeveLength.medium


def test_shape_attributes_are_restricted_to_their_slots() -> None:
    lower = parse("lower: skirt, black, pure color, cotton, long").garments[0]
    assert lower.length is LowerLength.long
    assert lower.sleeve_length is None

    footwear = parse("footwear: boots, black, pure color, leather, na").garments[0]
    assert footwear.sleeve_length is None
    assert footwear.length is None
    assert footwear.neckline is None


def test_dress_takes_both_sleeve_and_hem_length() -> None:
    garment = parse("dress: gown, red, floral, chiffon, long sleeves, maxi").garments[0]
    assert garment.sleeve_length is SleeveLength.long
    assert garment.length is LowerLength.long
    assert garment.neckline is Neckline.na


def test_unmatched_applicable_shape_attributes_become_na() -> None:
    garment = parse("upper: tee, white, graphic, cotton").garments[0]
    assert garment.sleeve_length is SleeveLength.na
    assert garment.neckline is Neckline.na
    assert garment.length is None


def test_attributes_are_read_from_the_item_field() -> None:
    garment = parse("lower: blue denim jeans").garments[0]
    assert garment.category == "blue denim jeans"
    assert garment.color is ColorName.blue
    assert garment.fabric is Fabric.denim


def test_unknown_attribute_values_fall_back_to_defaults() -> None:
    description = parse("upper: tee, ultraviolet, paisley, neoprene, na")
    garment = description.garments[0]
    assert len(description.garments) == 1
    assert garment.category == "tee"
    assert garment.color is ColorName.unknown
    assert garment.pattern is Pattern.na
    assert garment.fabric is Fabric.na


def test_unknown_slot_keys_are_skipped() -> None:
    description = parse(
        "accessory: watch, gold, na, na, na\nupper: tee, white, graphic, cotton, na"
    )
    assert len(description.garments) == 1
    assert description.garments[0].slot is GarmentSlot.upper


@pytest.mark.parametrize(
    "line",
    [
        "- upper: tee, white, graphic, cotton, na",
        "1. upper: tee, white, graphic, cotton, na",
        "2) upper: tee, white, graphic, cotton, na",
        "\u2022 upper: tee, white, graphic, cotton, na",
        "\u2014 upper: tee, white, graphic, cotton, na",
        "> upper: tee, white, graphic, cotton, na",
        "**upper:** tee, white, graphic, cotton, na",
        "`upper`: tee, white, graphic, cotton, na",
        "## upper: tee, white, graphic, cotton, na",
        "Assistant: upper: tee, white, graphic, cotton, na",
        "upper = tee, white, graphic, cotton, na",
    ],
)
def test_line_decorations_are_stripped(line: str) -> None:
    description = parse(line)
    assert len(description.garments) == 1
    assert description.garments[0].slot is GarmentSlot.upper
    assert description.garments[0].category == "tee"


def test_several_decorated_lines_parse_together() -> None:
    description = parse(
        "- upper: tee, white, graphic, cotton, na\n"
        "1. lower: jeans, blue, pure color, denim, long"
    )
    assert len(description.garments) == 2


def test_pipe_separated_fields_parse() -> None:
    garment = parse("upper: tee | white | graphic | cotton | na").garments[0]
    assert garment.color is ColorName.white
    assert garment.pattern is Pattern.graphic
    assert garment.fabric is Fabric.cotton


def test_semicolons_separate_records() -> None:
    description = parse(
        "upper: tee, white, graphic, cotton, na; lower: jeans, blue, pure color, denim, long"
    )
    assert len(description.garments) == 2


def test_prose_lines_are_skipped() -> None:
    description = parse(
        "The person stands outdoors on a bright afternoon.\n"
        "They look relaxed and comfortable in the photograph.\n"
        "Behind them a brick wall runs the width of the frame.\n"
        "Overall the mood is casual.\n"
        "upper: tee, white, graphic, cotton, na"
    )
    assert len(description.garments) == 1


def test_category_falls_back_to_the_slot_name() -> None:
    assert parse("upper:").garments[0].category == "upper"
    garment = parse("upper: , white, graphic, cotton, na").garments[0]
    assert garment.category == "upper"
    assert garment.color is ColorName.white


def test_leading_vocabulary_word_is_not_a_category() -> None:
    garment = parse("upper: black, plain, cotton, na, na").garments[0]
    assert garment.category == "upper"
    assert garment.color is ColorName.black
    assert garment.pattern is Pattern.pure_color


def test_inapplicable_vocabulary_word_stays_a_category() -> None:
    garment = parse("neckwear: collar, red, pure color, cotton").garments[0]
    assert garment.category == "collar"
    assert garment.color is ColorName.red


def test_category_is_truncated_to_the_schema_limit() -> None:
    assert len(parse(f"upper: {'x' * 200}").garments[0].category) == 64


def test_repeated_slots_get_numbered_refs() -> None:
    description = parse(
        "upper: tee, white, graphic, cotton, na\n"
        "upper: cardigan, black, pure color, knitted, long sleeves"
    )
    assert description.refs() == ["upper_0", "upper_1"]
    assert {garment.category for garment in description.garments} == {"tee", "cardigan"}


def test_identical_repeated_lines_collapse_to_one_garment() -> None:
    line = "upper: tee, white, graphic, cotton, na"
    assert len(parse("\n".join([line] * 5)).garments) == 1


def test_repetition_differing_only_in_case_collapses() -> None:
    description = parse(
        "upper: Tee, white, graphic, cotton, na\nupper: tee, white, graphic, cotton, na"
    )
    assert len(description.garments) == 1


def test_four_jewelry_lines_number_from_zero() -> None:
    description = parse(
        "jewelry: necklace, gold, na, na, na\n"
        "jewelry: bracelet, silver, na, na, na\n"
        "jewelry: ring, gold, na, na, na\n"
        "jewelry: earrings, silver, na, na, na"
    )
    assert description.refs() == ["jewelry_0", "jewelry_1", "jewelry_2", "jewelry_3"]


def test_empty_output_yields_the_fallback() -> None:
    description = parse("")
    garment = description.garments[0]
    assert len(description.garments) == 1
    assert garment.slot is GarmentSlot.other
    assert garment.category == "unknown"
    assert garment.confidence == 0.0
    assert garment.color is ColorName.unknown
    assert garment.pattern is Pattern.na
    assert garment.fabric is Fabric.na
    assert description.refs() == ["other_0"]


def test_whitespace_only_output_yields_the_fallback() -> None:
    description = parse("\n \t\r\n")
    assert description.garments[0].category == "unknown"
    assert description.caption == ""


def test_garbage_output_yields_the_fallback() -> None:
    garment = parse("@@@ ###\n%%% ^^^\n:::").garments[0]
    assert garment.slot is GarmentSlot.other
    assert garment.confidence == 0.0


def test_fallback_shape_fields_are_none() -> None:
    garment = parse("").garments[0]
    assert garment.sleeve_length is None
    assert garment.length is None
    assert garment.neckline is None
    assert garment.color_lab is None
    assert garment.color_lab_source is None
    assert garment.area_fraction is None


def test_json_output_yields_the_fallback() -> None:
    garment = parse('{"upper": {"color": "white"}}').garments[0]
    assert garment.slot is GarmentSlot.other
    assert garment.confidence == 0.0


def test_truncated_output_keeps_the_complete_records() -> None:
    description = parse("upper: tee, white, graphic, cotton, na\nlower: jea")
    assert len(description.garments) == 2
    assert description.garments[1].category == "jea"
    assert description.garments[1].pattern is Pattern.na


def test_truncated_tail_without_a_colon_is_dropped() -> None:
    assert len(parse("upper: tee, white, graphic, cotton, na\nlow").garments) == 1


@pytest.mark.parametrize(
    "text",
    [
        "\x00\x01",
        "," * 10000,
        ":",
        "::::",
        "\U0001f9e5\U0001f457\U0001f45f",
        "x" * 3000,
        "upper" * 500,
        "None",
        "-" * 200,
        "\x1b[31mupper\x1b[0m",
        "\ud800",
        "upper: \ud800\udfff",
    ],
)
def test_arbitrary_text_never_raises(text: str) -> None:
    description = parse(text)
    assert isinstance(description, OutfitDescription)
    assert len(description.garments) >= 1


def test_truncation_limits_track_the_schema() -> None:
    assert postprocess._MAX_CATEGORY == max_length(Garment, "category")
    assert postprocess._MAX_CAPTION == max_length(OutfitDescription, "caption")


def test_attribute_groups_partition_the_vocabulary() -> None:
    shape = {name for names in postprocess._SHAPE_ATTRIBUTES.values() for name in names}
    universal = set(postprocess._UNIVERSAL_ATTRIBUTES)
    assert universal | shape == {name for name, _ in postprocess._ATTRIBUTES}
    assert not universal & shape


def test_prose_names_garments_when_no_slot_key_is_present() -> None:
    description = parse(
        "The woman wears a dark blue dress with a pleated skirt. "
        "She holds a black clutch purse in her right hand."
    )
    assert [garment.slot for garment in description.garments] == [
        GarmentSlot.dress,
        GarmentSlot.lower,
        GarmentSlot.bag,
    ]
    assert description.garments[0].category == "dark blue dress"
    assert description.garments[0].color is ColorName.navy


def test_prose_splits_on_layering_words() -> None:
    description = parse("She is wearing a white cardigan over a grey shirt and black pants.")
    assert [(g.slot, g.category) for g in description.garments] == [
        (GarmentSlot.outer, "white cardigan"),
        (GarmentSlot.upper, "grey shirt"),
        (GarmentSlot.lower, "black pants"),
    ]
    assert description.garments[1].color is ColorName.gray


def test_prose_drops_possessives_from_the_category() -> None:
    garment = parse("The man's boots are brown.").garments[0]
    assert garment.slot is GarmentSlot.footwear
    assert garment.category == "boots"


def test_prose_garments_keep_full_confidence() -> None:
    description = parse("A man in a navy blazer.")
    assert description.garments[0].slot is GarmentSlot.outer
    assert description.garments[0].confidence == 1.0


def test_prose_without_any_garment_word_still_falls_back() -> None:
    garment = parse("The woman has a mask on her face.").garments[0]
    assert garment.slot is GarmentSlot.other
    assert garment.confidence == 0.0


def test_an_echoed_vocabulary_list_is_not_a_description() -> None:
    description = parse(
        "slot: upper, outer, lower, dress, romper, footwear, headwear, bag\n"
        "color: red, orange, yellow, green, blue, black, white"
    )
    assert [garment.slot for garment in description.garments] == [GarmentSlot.other]
    assert description.garments[0].confidence == 0.0


def test_slot_lines_win_over_prose_mining() -> None:
    description = parse(
        "She is wearing a white cardigan and black pants.\n"
        "upper: tee, white, graphic, cotton, na"
    )
    assert [garment.slot for garment in description.garments] == [GarmentSlot.upper]
    assert description.garments[0].category == "tee"


def test_signature_is_stable() -> None:
    def parse_description_reference(
        text: str,
        *,
        image_id: str,
        source_model: str,
        image_path: str = "",
        config_path: Path | None = None,
    ) -> OutfitDescription: ...

    assert inspect.signature(parse_description) == inspect.signature(parse_description_reference)


def test_public_surface_is_minimal() -> None:
    assert postprocess.__all__ == ["parse_description"]
