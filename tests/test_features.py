"""Cover the shared outfit feature vector and its saved contract."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from itertools import groupby
from typing import Any

import numpy as np
import pytest

from conftest import EXPECTED_IDS
from rmo.config import ConfigError, load_scoring_config
from rmo.schemas import (
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
from rmo.scoring import features
from rmo.scoring.features import (
    FeatureSpec,
    build_spec,
    describe_to_features,
    feature_contract,
    feature_names,
    select_slots,
)
from rmo.scoring.rules import largest_measured

CANONICAL_SLOTS = (
    GarmentSlot.upper,
    GarmentSlot.outer,
    GarmentSlot.lower,
    GarmentSlot.dress,
    GarmentSlot.footwear,
)

SAMPLE_ID = "fixture_000"
DUPLICATE_SLOT_ID = "fx_deg_06"
ALL_NONE_ID = "fx_deg_05"

PER_SLOT_BLOCKS = 10

PER_SLOT_WIDTH = (
    1
    + len(ColorName)
    + 4
    + len(Pattern)
    + len(Fabric)
    + len(SleeveLength)
    + 1
    + len(LowerLength)
    + 1
    + len(Neckline)
    + 1
    + 2
    + 2
    + 4
)

AGGREGATE_WIDTH = 12 + 4 + 4 + 5 + 2

DIGEST = re.compile(r"[0-9a-f]{64}")

RED_LAB = (53.24, 80.09, 67.20)
NAVY_LAB = (20.17, 23.45, -44.41)
WHITE_LAB = (94.73, 0.23, -0.25)


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return build_spec()


def base_config() -> dict[str, Any]:
    """Return a mutable copy of the shipped scoring configuration."""
    return copy.deepcopy(load_scoring_config())


def garment(slot: GarmentSlot, ref: str, category: str = "t-shirt", **fields: Any) -> Garment:
    """Return one garment with the supplied overrides."""
    return Garment(ref=ref, slot=slot, category=category, **fields)


def described(*garments: Garment) -> OutfitDescription:
    """Return a description wrapping the supplied garments."""
    return OutfitDescription(
        image_id="synthetic",
        source_model="test",
        garments=list(garments),
        provenance=Provenance.fixture,
    )


def base_garments() -> list[Garment]:
    """Return a measured outfit covering three in-scope slots."""
    return [
        garment(
            GarmentSlot.upper,
            "upper_0",
            "t-shirt",
            color=ColorName.red,
            color_lab=RED_LAB,
            area_fraction=0.4,
            pattern=Pattern.pure_color,
            fabric=Fabric.cotton,
            sleeve_length=SleeveLength.short,
            neckline=Neckline.round,
        ),
        garment(
            GarmentSlot.lower,
            "lower_0",
            "jeans",
            color=ColorName.navy,
            color_lab=NAVY_LAB,
            area_fraction=0.4,
            pattern=Pattern.pure_color,
            fabric=Fabric.denim,
            length=LowerLength.long,
        ),
        garment(
            GarmentSlot.footwear,
            "footwear_0",
            "sneakers",
            color=ColorName.white,
            color_lab=WHITE_LAB,
            area_fraction=0.2,
            fabric=Fabric.other,
        ),
    ]


def value_of(vector: np.ndarray, spec: FeatureSpec, name: str) -> float:
    """Return one named column of a feature vector."""
    return float(vector[spec.names.index(name)])


def changed(spec: FeatureSpec, before: np.ndarray, after: np.ndarray) -> set[str]:
    """Return the names of the columns whose value differs between two vectors."""
    return {name for index, name in enumerate(spec.names) if before[index] != after[index]}


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_every_fixture_yields_a_finite_fixed_width_vector_with_binary_indicators(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    spec: FeatureSpec,
) -> None:
    names = feature_names(spec)
    vector = describe_to_features(descriptions[image_id], spec)

    assert vector.dtype == np.float64
    assert vector.ndim == 1
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))

    indicators = [index for index, name in enumerate(names) if name in spec.indicator_names]
    assert indicators
    assert set(np.unique(vector[indicators])) <= {0.0, 1.0}


def test_the_vector_width_follows_the_block_table_and_the_schema_enum_sizes(
    spec: FeatureSpec,
) -> None:
    assert len(feature_names(spec)) == len(spec.slots) * PER_SLOT_WIDTH + AGGREGATE_WIDTH


def test_feature_names_are_unique_repeatable_and_equal_across_two_builds() -> None:
    first = build_spec(base_config())
    second = build_spec(base_config())
    names = feature_names(first)

    assert len(names) == len(set(names))
    assert names == feature_names(second)
    assert first.indicator_names == second.indicator_names


def test_two_specifications_share_no_mutable_table() -> None:
    first = build_spec(base_config())
    second = build_spec(base_config())

    assert first.formality.levels is not second.formality.levels
    assert first.season.fabric_warmth is not second.season.fabric_warmth
    assert first.formality.levels == second.formality.levels

    levels: Any = first.formality.levels
    warmths: Any = first.season.fabric_warmth
    with pytest.raises(TypeError):
        levels["blazer"] = 0
    with pytest.raises(TypeError):
        warmths[Fabric.denim] = 0.0


def test_slot_columns_follow_the_canonical_upper_outer_lower_dress_footwear_order(
    spec: FeatureSpec,
) -> None:
    assert spec.slots == CANONICAL_SLOTS

    slot_values = {slot.value for slot in spec.slots}
    prefixes = [name.split("__", 1)[0] for name in feature_names(spec)]
    runs = [prefix for prefix, _ in groupby(prefixes) if prefix in slot_values]

    assert runs == [slot.value for slot in spec.slots] * PER_SLOT_BLOCKS


def test_permuting_the_garment_list_with_stable_refs_changes_nothing(
    descriptions: dict[str, OutfitDescription], spec: FeatureSpec
) -> None:
    original = descriptions[DUPLICATE_SLOT_ID]
    shuffled = original.model_copy(deep=True)
    shuffled.garments = list(reversed(shuffled.garments))

    assert np.array_equal(
        describe_to_features(original, spec), describe_to_features(shuffled, spec)
    )


def test_the_selector_matches_the_rule_helper_on_the_duplicate_slot_fixture(
    descriptions: dict[str, OutfitDescription], spec: FeatureSpec
) -> None:
    description = descriptions[DUPLICATE_SLOT_ID]
    selected = select_slots(description, spec)

    assert set(selected) == set(spec.slots)
    for slot in spec.slots:
        assert selected[slot] is largest_measured(description.by_slot(slot))


@pytest.mark.parametrize(
    ("first", "second", "winner"),
    [
        (("a", 0.4), ("b", 0.1), "a"),
        (("a", 0.1), ("b", 0.4), "b"),
        (("a", 0.3), ("b", 0.3), "a"),
        (("b", 0.3), ("a", 0.3), "a"),
        (("a", 0.0), ("b", None), "a"),
        (("b", 0.0), ("a", None), "b"),
        (("a", None), ("b", None), "a"),
        (("b", None), ("a", None), "a"),
    ],
)
def test_the_selector_and_the_rule_helper_choose_the_same_representative(
    first: tuple[str, float | None],
    second: tuple[str, float | None],
    winner: str,
    spec: FeatureSpec,
) -> None:
    description = described(
        garment(GarmentSlot.upper, first[0], area_fraction=first[1]),
        garment(GarmentSlot.upper, second[0], area_fraction=second[1]),
    )
    chosen = select_slots(description, spec)[GarmentSlot.upper]

    assert chosen is largest_measured(description.by_slot(GarmentSlot.upper))
    assert chosen is not None
    assert chosen.ref == winner


@pytest.mark.parametrize("areas", [(0.3, 0.3), (0.4, 0.1)])
def test_changing_confidence_never_flips_the_selected_representative(
    areas: tuple[float, float], spec: FeatureSpec
) -> None:
    description = described(
        garment(GarmentSlot.upper, "a", area_fraction=areas[0], confidence=0.05),
        garment(GarmentSlot.upper, "b", area_fraction=areas[1], confidence=1.0),
    )
    chosen = select_slots(description, spec)[GarmentSlot.upper]

    assert chosen is not None
    assert chosen.ref == "a"


def test_adding_or_removing_a_losing_in_scope_duplicate_changes_no_value(
    spec: FeatureSpec,
) -> None:
    loser = garment(
        GarmentSlot.upper,
        "upper_9",
        "blazer",
        color=ColorName.green,
        color_lab=(87.7, -86.2, 83.2),
        area_fraction=0.05,
        pattern=Pattern.floral,
        fabric=Fabric.furry,
        sleeve_length=SleeveLength.sleeveless,
        neckline=Neckline.lapel,
    )
    without = describe_to_features(described(*base_garments()), spec)
    with_loser = describe_to_features(described(*base_garments(), loser), spec)

    assert np.array_equal(without, with_loser)


@pytest.mark.parametrize(
    ("slot", "category"),
    [
        (GarmentSlot.romper, "romper"),
        (GarmentSlot.jewelry, "hoop earrings"),
        (GarmentSlot.other, "unknown"),
    ],
)
def test_adding_an_out_of_scope_garment_changes_only_the_ignored_count(
    slot: GarmentSlot, category: str, spec: FeatureSpec
) -> None:
    extra = garment(
        slot,
        "extra_0",
        category,
        color=ColorName.green,
        color_lab=(87.7, -86.2, 83.2),
        area_fraction=0.3,
        pattern=Pattern.floral,
        fabric=Fabric.furry,
        sleeve_length=SleeveLength.sleeveless,
        length=LowerLength.three_point,
        neckline=Neckline.square,
    )
    before = describe_to_features(described(*base_garments()), spec)
    after = describe_to_features(described(*base_garments(), extra), spec)

    assert changed(spec, before, after) == {"composition__n_ignored"}
    assert value_of(after, spec, "composition__n_ignored") == 1.0
    assert value_of(before, spec, "composition__n_ignored") == 0.0


def upper_with(**fields: Any) -> OutfitDescription:
    """Return a one-upper outfit carrying the supplied overrides."""
    defaults: dict[str, Any] = {
        "color": ColorName.red,
        "color_lab": RED_LAB,
        "area_fraction": 0.25,
        "sleeve_length": SleeveLength.short,
    }
    return described(garment(GarmentSlot.upper, "upper_0", **{**defaults, **fields}))


def test_known_unknown_labelled_na_none_absent_and_measured_zero_encode_distinctly(
    spec: FeatureSpec,
) -> None:
    cases = {
        "known": upper_with(),
        "unknown_color": upper_with(color=ColorName.unknown),
        "labelled_na": upper_with(sleeve_length=SleeveLength.na),
        "nullable_none": upper_with(sleeve_length=None),
        "measured_zero": upper_with(area_fraction=0.0),
        "unmeasured": upper_with(area_fraction=None),
        "absent": described(
            garment(GarmentSlot.lower, "lower_0", "jeans", area_fraction=0.25)
        ),
    }
    vectors = {name: describe_to_features(case, spec) for name, case in cases.items()}
    packed = {name: vector.tobytes() for name, vector in vectors.items()}
    assert len(set(packed.values())) == len(cases)

    known = vectors["known"]
    assert value_of(known, spec, "upper__color__red") == 1.0
    assert value_of(known, spec, "upper__color__unknown") == 0.0
    assert value_of(known, spec, "upper__sleeve_length__short") == 1.0
    assert value_of(known, spec, "upper__sleeve_length__is_none") == 0.0

    unknown = vectors["unknown_color"]
    assert value_of(unknown, spec, "upper__color__unknown") == 1.0
    assert value_of(unknown, spec, "upper__color__red") == 0.0
    assert value_of(unknown, spec, "upper__slot_missing") == 0.0

    labelled = vectors["labelled_na"]
    assert value_of(labelled, spec, "upper__sleeve_length__na") == 1.0
    assert value_of(labelled, spec, "upper__sleeve_length__is_none") == 0.0

    nullable = vectors["nullable_none"]
    assert value_of(nullable, spec, "upper__sleeve_length__is_none") == 1.0
    assert all(
        value_of(nullable, spec, f"upper__sleeve_length__{member.value}") == 0.0
        for member in SleeveLength
    )

    zero = vectors["measured_zero"]
    assert value_of(zero, spec, "upper__area_fraction") == 0.0
    assert value_of(zero, spec, "upper__area_missing") == 0.0

    unmeasured = vectors["unmeasured"]
    assert value_of(unmeasured, spec, "upper__area_fraction") == 0.0
    assert value_of(unmeasured, spec, "upper__area_missing") == 1.0


def test_an_absent_slot_zeroes_every_field_indicator_except_slot_missing(
    spec: FeatureSpec,
) -> None:
    vector = describe_to_features(described(*base_garments()), spec)

    for name in feature_names(spec):
        if not name.startswith(f"{GarmentSlot.dress.value}__"):
            continue
        expected = 1.0 if name == "dress__slot_missing" else 0.0
        assert value_of(vector, spec, name) == expected


def test_a_non_finite_measured_lab_is_sentinelled_instead_of_propagated(
    spec: FeatureSpec,
) -> None:
    description = described(
        garment(
            GarmentSlot.upper,
            "upper_0",
            color=ColorName.red,
            color_lab=(float("nan"), 80.09, float("inf")),
            area_fraction=0.5,
        ),
        garment(
            GarmentSlot.lower,
            "lower_0",
            "jeans",
            color=ColorName.blue,
            color_lab=NAVY_LAB,
            area_fraction=0.5,
        ),
    )
    vector = describe_to_features(description, spec)

    assert np.all(np.isfinite(vector))
    assert value_of(vector, spec, "upper__color_lab_missing") == 1.0
    assert [value_of(vector, spec, f"upper__color_lab_{axis}") for axis in "lab"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert value_of(vector, spec, "lower__color_lab_missing") == 0.0


def test_an_undefined_aggregate_is_sentinelled_with_its_explicit_count(
    descriptions: dict[str, OutfitDescription], spec: FeatureSpec
) -> None:
    vector = describe_to_features(descriptions[ALL_NONE_ID], spec)

    assert value_of(vector, spec, "color__hue_delta_undefined") == 1.0
    assert value_of(vector, spec, "color__hue_delta_mean") == 0.0
    assert value_of(vector, spec, "color__hue_delta_max") == 0.0
    assert value_of(vector, spec, "color__hue_pair_count") == 0.0
    assert value_of(vector, spec, "color__lightness_contrast_undefined") == 1.0
    assert value_of(vector, spec, "color__lightness_contrast") == 0.0
    assert value_of(vector, spec, "formality__undefined") == 1.0
    assert value_of(vector, spec, "formality__mapped_count") == 0.0
    assert value_of(vector, spec, "season__undefined") == 1.0
    assert value_of(vector, spec, "proportion__upper_lower_ratio_undefined") == 1.0
    assert value_of(vector, spec, "proportion__measured_count") == 0.0


def test_a_reference_lab_fallback_reaches_the_aggregates_without_filling_the_measured_block(
    spec: FeatureSpec,
) -> None:
    description = described(
        garment(GarmentSlot.upper, "upper_0", color=ColorName.red, area_fraction=0.5),
        garment(
            GarmentSlot.lower, "lower_0", "jeans", color=ColorName.blue, area_fraction=0.5
        ),
    )
    vector = describe_to_features(description, spec)

    assert value_of(vector, spec, "upper__color_lab_missing") == 1.0
    assert value_of(vector, spec, "lower__color_lab_missing") == 1.0
    assert [value_of(vector, spec, f"upper__color_lab_{axis}") for axis in "lab"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert value_of(vector, spec, "color__hue_delta_undefined") == 0.0
    assert value_of(vector, spec, "color__hue_pair_count") == 1.0
    assert value_of(vector, spec, "color__hue_delta_max") > 0.0
    assert value_of(vector, spec, "color__lightness_contrast") > 0.0
    assert value_of(vector, spec, "color__lightness_contrast_undefined") == 0.0


def test_altering_excluded_metadata_changes_no_feature(
    descriptions: dict[str, OutfitDescription], spec: FeatureSpec
) -> None:
    original = descriptions[SAMPLE_ID]
    before = describe_to_features(original, spec)

    altered = original.model_copy(deep=True)
    altered.image_id = "some-other-image"
    altered.image_path = "data/elsewhere/other.png"
    altered.caption = "an entirely different caption"
    altered.provenance = Provenance.gt
    altered.source_model = "some-other-model"
    for item in altered.garments:
        item.confidence = 0.01
        item.color_lab_source = "wholeimage"

    assert np.array_equal(before, describe_to_features(altered, spec))


def test_extraction_reads_no_configuration_and_mutates_nothing(
    descriptions: dict[str, OutfitDescription],
    spec: FeatureSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    description = descriptions[SAMPLE_ID].model_copy(deep=True)
    snapshot = description.model_dump()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("extraction must not read the scoring configuration")

    for module in (
        "rmo.config",
        "rmo.scoring.color",
        "rmo.scoring.features",
        "rmo.scoring.palette",
        "rmo.scoring.rules",
    ):
        monkeypatch.setattr(f"{module}.load_scoring_config", refuse)

    vector = describe_to_features(description, spec)

    assert vector.shape == (len(spec.names),)
    assert description.model_dump() == snapshot


def test_an_unknown_block_name_is_rejected() -> None:
    config = base_config()
    config["features"]["blocks"] = [*config["features"]["blocks"], "slot_vibes"]
    with pytest.raises(ConfigError, match="slot_vibes"):
        build_spec(config)


def test_a_repeated_block_name_is_rejected() -> None:
    config = base_config()
    config["features"]["blocks"] = ["slot_presence", "slot_presence"]
    with pytest.raises(ConfigError, match="repeats 'slot_presence'"):
        build_spec(config)


def test_an_unknown_slot_name_is_rejected() -> None:
    config = base_config()
    config["features"]["slots"] = ["upper", "torso"]
    with pytest.raises(ConfigError, match="torso"):
        build_spec(config)


def test_a_repeated_slot_is_rejected() -> None:
    config = base_config()
    config["features"]["slots"] = ["upper", "upper"]
    with pytest.raises(ConfigError, match="repeats 'upper'"):
        build_spec(config)


def test_two_blocks_that_produce_the_same_column_name_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        features._HANDLERS, "echo_presence", features._HANDLERS["slot_presence"]
    )
    config = base_config()
    config["features"]["blocks"] = ["slot_presence", "echo_presence"]
    with pytest.raises(ConfigError, match="duplicate column 'upper__slot_missing'"):
        build_spec(config)


def test_the_contract_carries_the_layout_the_loader_has_to_verify(spec: FeatureSpec) -> None:
    contract = feature_contract(spec)

    assert set(contract) == {
        "feature_version",
        "feature_names",
        "slot_order",
        "blocks",
        "vocabularies",
        "dtype",
        "sentinel",
        "indicator_names",
        "selection",
        "settings",
        "spec_sha256",
    }
    assert contract["feature_names"] == feature_names(spec)
    assert contract["slot_order"] == [slot.value for slot in spec.slots]
    assert contract["dtype"] == "float64"
    assert contract["sentinel"] == 0.0
    assert contract["indicator_names"] == sorted(spec.indicator_names)
    assert contract["selection"] == {
        "strategy": "largest_measured_area",
        "tie_break": "ref",
    }
    assert contract["vocabularies"]["color"] == [member.value for member in ColorName]


def test_the_contract_digest_is_sixty_four_lowercase_hex_characters(
    spec: FeatureSpec,
) -> None:
    assert DIGEST.fullmatch(feature_contract(spec)["spec_sha256"])


def test_the_contract_digest_is_recomputable_from_the_contract_alone(
    spec: FeatureSpec,
) -> None:
    contract = feature_contract(spec)
    payload = {key: value for key, value in contract.items() if key != "spec_sha256"}
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == contract["spec_sha256"]


def test_the_contract_digest_is_equal_for_two_builds_of_the_same_configuration() -> None:
    first = feature_contract(build_spec(base_config()))
    second = feature_contract(build_spec(base_config()))

    assert first["spec_sha256"] == second["spec_sha256"]


def bump_version(config: dict[str, Any]) -> None:
    config["features"]["version"] = "2.0.0"


def reverse_slots(config: dict[str, Any]) -> None:
    config["features"]["slots"] = list(reversed(config["features"]["slots"]))


def rotate_blocks(config: dict[str, Any]) -> None:
    blocks = config["features"]["blocks"]
    config["features"]["blocks"] = [blocks[-1], *blocks[:-1]]


def widen_analogous_tolerance(config: dict[str, Any]) -> None:
    config["color"]["analogous_max"] = 25.0


@pytest.mark.parametrize(
    "mutate", [bump_version, reverse_slots, rotate_blocks, widen_analogous_tolerance]
)
def test_the_contract_digest_changes_when_the_specification_changes(
    mutate: Any,
) -> None:
    reference = feature_contract(build_spec(base_config()))["spec_sha256"]
    config = base_config()
    mutate(config)

    assert feature_contract(build_spec(config))["spec_sha256"] != reference
