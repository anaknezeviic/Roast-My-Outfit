"""Cover colour-theory relations between the garments of one outfit."""

from __future__ import annotations

import math

import pytest

from conftest import EXPECTED_IDS
from rmo.config import ConfigError
from rmo.scoring.color import (
    Relation,
    RelationSettings,
    chroma,
    chroma_contrast,
    distinct_hue_count,
    hue_angle,
    hue_pair_deltas,
    lightness_contrast,
    neutral_count,
    pairwise_hue_deltas,
    relation_of,
    relation_settings,
    resolved_lab,
    triadic_groups,
    violates_three_color_rule,
)
from rmo.scoring.palette import reference_lab
from rmo.schemas import ColorName, Garment, GarmentSlot, OutfitDescription

RED_GREEN = 96.017
GREEN_BLUE = 170.269
RED_BLUE = 93.714

SHIPPED_SECTION = {
    "achromatic_chroma": 5.0,
    "analogous_max": 30.0,
    "complementary_tolerance": 20.0,
    "triadic_tolerance": 20.0,
    "hue_separation": 30.0,
    "max_hue_families": 3,
}


@pytest.fixture(scope="module")
def settings() -> RelationSettings:
    return relation_settings()


def lab_at(hue_deg: float, *, chroma: float = 60.0, lightness: float = 50.0):
    """Return the Lab coordinates of one LCh position."""
    radians = math.radians(hue_deg)
    return (lightness, chroma * math.cos(radians), chroma * math.sin(radians))


def hued(ref: str, hue_deg: float, *, color: ColorName = ColorName.red) -> Garment:
    """Return a garment whose measured Lab sits at one hue angle."""
    return Garment(
        ref=ref,
        slot=GarmentSlot.upper,
        category="shirt",
        color=color,
        color_lab=lab_at(hue_deg),
    )


def named(ref: str, color: ColorName) -> Garment:
    """Return a garment that carries only a colour name."""
    return Garment(ref=ref, slot=GarmentSlot.upper, category="shirt", color=color)


def delta_between(first: Garment, second: Garment, settings: RelationSettings) -> float:
    """Return the single hue delta of one garment pair."""
    deltas = pairwise_hue_deltas([first, second], settings=settings)
    assert len(deltas) == 1
    return deltas[0]


def test_hues_half_a_turn_apart_are_complementary(settings: RelationSettings) -> None:
    delta = delta_between(hued("a", 0.0), hued("b", 180.0), settings)
    assert delta == pytest.approx(180.0)
    assert relation_of(delta, settings=settings) is Relation.complementary


def test_neighbouring_hues_are_analogous(settings: RelationSettings) -> None:
    delta = delta_between(hued("a", 0.0), hued("b", 20.0), settings)
    assert delta == pytest.approx(20.0)
    assert relation_of(delta, settings=settings) is Relation.analogous


def test_hues_across_the_wrap_boundary_stay_two_degrees_apart(
    settings: RelationSettings,
) -> None:
    delta = delta_between(hued("a", 359.0), hued("b", 1.0), settings)
    assert delta == pytest.approx(2.0)
    assert relation_of(delta, settings=settings) is Relation.analogous


def test_three_evenly_spaced_hues_form_exactly_one_triad(
    settings: RelationSettings,
) -> None:
    garments = [hued("a", 0.0), hued("b", 120.0), hued("c", 240.0)]
    assert triadic_groups(garments, settings=settings) == [("a", "b", "c")]


def test_a_pair_a_third_of_a_turn_apart_is_only_a_candidate(
    settings: RelationSettings,
) -> None:
    pair = [hued("a", 0.0), hued("b", 120.0)]
    delta = delta_between(pair[0], pair[1], settings)
    assert relation_of(delta, settings=settings) is Relation.triadic_candidate
    assert triadic_groups(pair, settings=settings) == []


def test_triadic_groups_are_permutation_invariant(settings: RelationSettings) -> None:
    garments = [hued("a", 0.0), hued("b", 120.0), hued("c", 240.0)]
    reversed_groups = triadic_groups(list(reversed(garments)), settings=settings)
    assert reversed_groups == triadic_groups(garments, settings=settings)


@pytest.mark.parametrize(
    ("first", "second", "expected", "relation"),
    [
        (ColorName.red, ColorName.green, RED_GREEN, Relation.none),
        (ColorName.green, ColorName.blue, GREEN_BLUE, Relation.complementary),
        (ColorName.red, ColorName.blue, RED_BLUE, Relation.none),
    ],
)
def test_reference_colours_produce_their_measured_lab_deltas(
    settings: RelationSettings,
    first: ColorName,
    second: ColorName,
    expected: float,
    relation: Relation,
) -> None:
    delta = delta_between(named("a", first), named("b", second), settings)
    assert delta == pytest.approx(expected, abs=1e-3)
    assert relation_of(delta, settings=settings) is relation


def test_an_all_neutral_outfit_has_no_hue_pairs_but_a_defined_chroma_spread(
    settings: RelationSettings,
) -> None:
    garments = [
        named("a", ColorName.black),
        named("b", ColorName.white),
        named("c", ColorName.navy),
        named("d", ColorName.beige),
    ]
    assert pairwise_hue_deltas(garments, settings=settings) == []
    assert distinct_hue_count(garments, settings=settings) == 0
    assert triadic_groups(garments, settings=settings) == []

    spread = chroma_contrast(garments, settings=settings)
    assert isinstance(spread, float)
    assert math.isfinite(spread)


def test_hue_pair_deltas_name_the_two_garments_of_every_pair(
    settings: RelationSettings,
) -> None:
    garments = [hued("a", 0.0), named("b", ColorName.black), hued("c", 120.0)]
    pairs = hue_pair_deltas(garments, settings=settings)

    assert [(first, second) for first, second, _ in pairs] == [("a", "c")]
    assert pairs[0][2] == pytest.approx(120.0)
    assert [delta for _, _, delta in pairs] == pairwise_hue_deltas(
        garments, settings=settings
    )


def test_a_single_garment_returns_a_defined_value_from_every_function(
    settings: RelationSettings,
) -> None:
    garments = [hued("a", 40.0)]
    assert pairwise_hue_deltas(garments, settings=settings) == []
    assert triadic_groups(garments, settings=settings) == []
    assert chroma_contrast(garments, settings=settings) == 0.0
    assert lightness_contrast(garments) == 0.0
    assert neutral_count(garments) == 0
    assert distinct_hue_count(garments, settings=settings) == 1
    assert violates_three_color_rule(garments, settings=settings) is False


def test_an_achromatic_lab_reports_no_hue_at_all(settings: RelationSettings) -> None:
    assert hue_angle((50.0, 0.0, 0.0), settings=settings) is None
    assert hue_angle((50.0, -0.0, -0.0), settings=settings) is None
    assert chroma((50.0, 0.0, 0.0)) == 0.0


def test_four_hue_families_break_the_three_colour_rule_and_three_do_not(
    settings: RelationSettings,
) -> None:
    three = [hued("a", 0.0), hued("b", 120.0), hued("c", 240.0)]
    assert distinct_hue_count(three, settings=settings) == 3
    assert violates_three_color_rule(three, settings=settings) is False

    four = [*three, hued("d", 60.0)]
    assert distinct_hue_count(four, settings=settings) == 4
    assert violates_three_color_rule(four, settings=settings) is True


def test_hue_families_survive_permutation_and_repeats(
    settings: RelationSettings,
) -> None:
    repeated = [hued("a", 0.0), hued("b", 0.0), hued("c", 0.0)]
    assert distinct_hue_count(repeated, settings=settings) == 1

    garments = [hued("a", 0.0), hued("b", 120.0), hued("c", 240.0)]
    shuffled = [garments[2], garments[0], garments[1]]
    assert distinct_hue_count(shuffled, settings=settings) == 3


def test_hue_families_do_not_split_across_the_wrap_boundary(
    settings: RelationSettings,
) -> None:
    assert distinct_hue_count([hued("a", 359.0), hued("b", 1.0)], settings=settings) == 1


@pytest.mark.parametrize(("count", "families"), [(3, 3), (13, 1), (20, 1)])
def test_evenly_spaced_hues_chain_into_one_family_below_the_separation(
    settings: RelationSettings, count: int, families: int
) -> None:
    step = 360.0 / count
    garments = [hued(f"g{index}", index * step) for index in range(count)]
    assert distinct_hue_count(garments, settings=settings) == families


def test_twelve_hues_at_the_exact_separation_still_occupy_a_family(
    settings: RelationSettings,
) -> None:
    garments = [
        hued(f"g{index}", index * settings.hue_separation) for index in range(12)
    ]
    assert 1 <= distinct_hue_count(garments, settings=settings) <= 12


def test_a_gap_of_exactly_the_separation_keeps_two_hues_in_one_family(
    settings: RelationSettings,
) -> None:
    exact = [hued("a", 0.0), hued("b", settings.hue_separation)]
    wider = [hued("a", 0.0), hued("b", settings.hue_separation + 0.0001)]

    assert distinct_hue_count(exact, settings=settings) == 1
    assert distinct_hue_count(wider, settings=settings) == 2


def test_a_named_colour_without_a_measured_lab_still_carries_a_hue(
    settings: RelationSettings,
) -> None:
    garment = named("a", ColorName.red)
    assert distinct_hue_count([garment], settings=settings) == 1
    assert resolved_lab(garment) == reference_lab(ColorName.red)
    assert garment.color_lab is None


def test_a_measured_lab_wins_over_the_named_reference() -> None:
    measured = lab_at(200.0)
    garment = Garment(
        ref="a",
        slot=GarmentSlot.upper,
        category="shirt",
        color=ColorName.red,
        color_lab=measured,
    )
    assert resolved_lab(garment) == pytest.approx(measured)
    assert resolved_lab(garment) != reference_lab(ColorName.red)


def test_neutral_count_ignores_unknown_colours() -> None:
    garments = [
        named("a", ColorName.black),
        named("b", ColorName.unknown),
        named("c", ColorName.brown),
        named("d", ColorName.red),
    ]
    assert neutral_count(garments) == 2


def test_an_unknown_colour_contributes_no_hue_and_no_lab(
    settings: RelationSettings,
) -> None:
    garment = named("a", ColorName.unknown)
    assert resolved_lab(garment) is None
    assert distinct_hue_count([garment], settings=settings) == 0
    assert lightness_contrast([garment, named("b", ColorName.red)]) == 0.0


@pytest.mark.parametrize("bad", [(float("nan"), 1.0, 2.0), (50.0, float("inf"), 0.0), (50.0, 0.0)])
def test_a_malformed_lab_tuple_is_rejected(
    settings: RelationSettings, bad: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="three finite numbers"):
        chroma(bad)
    with pytest.raises(ValueError, match="three finite numbers"):
        hue_angle(bad, settings=settings)


@pytest.mark.parametrize("bad", [(1, True, False), (True, 0.0, 0.0), (50.0, 0.0, True)])
def test_a_boolean_lab_component_is_rejected(
    settings: RelationSettings, bad: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError, match="three finite numbers"):
        chroma(bad)
    with pytest.raises(ValueError, match="three finite numbers"):
        hue_angle(bad, settings=settings)


def test_a_non_finite_measured_lab_falls_back_without_mutating_the_description(
    settings: RelationSettings,
) -> None:
    broken = (float("nan"), 1.0, 2.0)
    description = OutfitDescription(
        image_id="broken",
        source_model="test",
        garments=[
            Garment(
                slot=GarmentSlot.upper,
                category="shirt",
                color=ColorName.red,
                color_lab=broken,
            )
        ],
    )
    garment = description.garments[0]

    assert resolved_lab(garment) == reference_lab(ColorName.red)
    assert distinct_hue_count(description.garments, settings=settings) == 1
    assert garment.color_lab is not None
    assert math.isnan(garment.color_lab[0])
    assert garment.color_lab[1:] == broken[1:]


@pytest.mark.parametrize("bad_delta", [float("nan"), -1.0, 180.5])
def test_a_delta_outside_the_folded_range_is_rejected(
    settings: RelationSettings, bad_delta: float
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        relation_of(bad_delta, settings=settings)


def test_reference_lab_lands_on_the_independent_achromatic_anchors() -> None:
    assert reference_lab(ColorName.white) == pytest.approx((100.0, 0.0, 0.0), abs=0.05)
    assert reference_lab(ColorName.black) == pytest.approx((0.0, 0.0, 0.0), abs=0.05)

    gray = reference_lab(ColorName.gray)
    assert gray is not None
    assert gray[1:] == pytest.approx((0.0, 0.0), abs=0.05)
    assert 0.0 < gray[0] < 100.0

    assert reference_lab(ColorName.unknown) is None


def test_the_shipped_relation_bands_do_not_overlap(settings: RelationSettings) -> None:
    assert settings.analogous_max < 120.0 - settings.triadic_tolerance
    assert 120.0 + settings.triadic_tolerance < 180.0 - settings.complementary_tolerance


def test_complementary_outranks_analogous_when_the_bands_overlap() -> None:
    overlapping = RelationSettings(
        achromatic_chroma=5.0,
        analogous_max=170.0,
        complementary_tolerance=20.0,
        triadic_tolerance=20.0,
        hue_separation=30.0,
        max_hue_families=3,
    )
    assert relation_of(175.0, settings=overlapping) is Relation.complementary
    assert relation_of(150.0, settings=overlapping) is Relation.analogous


@pytest.mark.parametrize("payload", [{}, {"color": None}, {"color": [1, 2]}])
def test_a_missing_colour_section_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ConfigError, match="no color section"):
        relation_settings(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"achromatic_chroma": -1.0},
        {"achromatic_chroma": float("nan")},
        {"analogous_max": 181.0},
        {"complementary_tolerance": 91.0},
        {"triadic_tolerance": 61.0},
        {"hue_separation": 181.0},
        {"hue_separation": "wide"},
        {"max_hue_families": 0},
        {"max_hue_families": True},
        {"max_hue_families": 2.5},
    ],
)
def test_an_out_of_range_colour_option_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        relation_settings({"color": {**SHIPPED_SECTION, **overrides}})


def test_the_shipped_section_is_accepted_verbatim() -> None:
    assert relation_settings({"color": dict(SHIPPED_SECTION)}) == relation_settings()


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_every_fixture_yields_typed_and_finite_relation_facts(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    settings: RelationSettings,
) -> None:
    garments = descriptions[image_id].garments

    deltas = pairwise_hue_deltas(garments, settings=settings)
    assert isinstance(deltas, list)
    for delta in deltas:
        assert isinstance(delta, float)
        assert math.isfinite(delta)
        assert 0.0 <= delta <= 180.0
        assert isinstance(relation_of(delta, settings=settings), Relation)

    groups = triadic_groups(garments, settings=settings)
    assert groups == sorted(groups)
    for group in groups:
        assert len(group) == 3
        assert all(ref in descriptions[image_id].refs() for ref in group)

    for spread in (
        chroma_contrast(garments, settings=settings),
        lightness_contrast(garments),
    ):
        assert isinstance(spread, float)
        assert math.isfinite(spread)

    for count in (neutral_count(garments), distinct_hue_count(garments, settings=settings)):
        assert count >= 0

    assert isinstance(violates_three_color_rule(garments, settings=settings), bool)

    for garment in garments:
        lab = resolved_lab(garment)
        if lab is None:
            continue
        value = chroma(lab)
        assert isinstance(value, float)
        assert math.isfinite(value)
        angle = hue_angle(lab, settings=settings)
        if angle is not None:
            assert isinstance(angle, float)
            assert math.isfinite(angle)
            assert 0.0 <= angle < 360.0
