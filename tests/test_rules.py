"""Cover the hand-authored rule scorer and its configuration tables."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest

from conftest import EXPECTED_IDS
from rmo.config import ConfigError, load_scoring_config
from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    Issue,
    IssueCode,
    IssueSeverity,
    LowerLength,
    OutfitDescription,
    OutfitScore,
    Pattern,
    Provenance,
    SleeveLength,
    SubScores,
)
from rmo.scoring import rules
from rmo.scoring.rules import (
    RuleScorer,
    _MESSAGES,
    category_key,
    formality_level,
    largest_measured,
    rule_settings,
)

SINGLE_GARMENT_ID = "fx_deg_00"
NO_FOOTWEAR_ID = "fx_deg_01"
ALL_NEUTRAL_ID = "fx_deg_03"
FOUR_JEWELLERY_ID = "fx_deg_04"
ALL_NONE_ID = "fx_deg_05"
FAULTLESS_ID = "fx_adv_02"
TWO_PATTERNS_ID = "fx_adv_00"

MESSAGE_KWARGS: dict[str, frozenset[str]] = {
    "hue_clash": frozenset({"a", "b", "delta"}),
    "too_many_colors": frozenset({"families", "limit"}),
    "low_contrast": frozenset({"a", "b", "spread"}),
    "monochrome_flat": frozenset({"spread"}),
    "pattern_clash": frozenset({"count", "limit"}),
    "formality_mismatch": frozenset({"a", "b", "gap"}),
    "season_mismatch": frozenset({"a", "b"}),
    "fabric_mismatch": frozenset({"a", "b"}),
    "dominant_area": frozenset({"a", "share"}),
    "upper_lower_ratio": frozenset({"a", "b", "ratio"}),
    "missing_footwear": frozenset(),
    "accessory_overload": frozenset({"count", "limit"}),
    "other": frozenset(),
}

ENUM_PREFIXES = tuple(
    f"{enum_type.__name__}."
    for enum_type in (
        ColorName,
        Fabric,
        GarmentSlot,
        IssueCode,
        IssueSeverity,
        LowerLength,
        Pattern,
        SleeveLength,
    )
)


@pytest.fixture(scope="module")
def scorer() -> RuleScorer:
    return RuleScorer()


def garment(slot: GarmentSlot, category: str = "shirt", **fields) -> Garment:
    """Return one garment with the supplied overrides."""
    return Garment(slot=slot, category=category, **fields)


def described(*garments: Garment) -> OutfitDescription:
    """Return a description wrapping the supplied garments."""
    return OutfitDescription(
        image_id="synthetic",
        source_model="test",
        garments=list(garments),
        provenance=Provenance.fixture,
    )


def codes(score: OutfitScore) -> set[IssueCode]:
    """Return the distinct issue codes of a score."""
    return {issue.code for issue in score.issues}


def lab_at(hue_deg: float, lightness: float) -> tuple[float, float, float]:
    """Return the Lab coordinates of one LCh position at full chroma."""
    radians = math.radians(hue_deg)
    return (lightness, 60.0 * math.cos(radians), 60.0 * math.sin(radians))


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_every_fixture_scores_within_bounds_with_grounded_neutral_messages(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scorer: RuleScorer,
) -> None:
    description = descriptions[image_id]
    score = scorer.score(description)

    assert score.image_id == description.image_id
    assert score.provenance == description.provenance
    assert score.source_model == RuleScorer.name

    for value in (score.overall, *score.subscores.model_dump().values()):
        assert isinstance(value, float)
        assert not math.isnan(value)
        assert 0.0 <= value <= 100.0

    refs = set(description.refs())
    for issue in score.issues:
        assert set(issue.garment_refs) <= refs
        assert 1 <= len(issue.message) <= 280
        assert "!" not in issue.message
        assert "_" not in issue.message
        assert not any(prefix in issue.message for prefix in ENUM_PREFIXES)

    assert not {IssueCode.low_contrast, IssueCode.monochrome_flat} <= codes(score)


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_scoring_repeats_exactly_and_leaves_the_description_unchanged(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scorer: RuleScorer,
) -> None:
    description = descriptions[image_id]
    before = description.model_dump_json()

    assert scorer.score(description) == scorer.score(description)
    assert description.model_dump_json() == before


def test_the_scorer_announces_the_frozen_model_name() -> None:
    assert RuleScorer.name == "rule_scorer_v1"


def test_a_single_garment_scores_without_raising(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    score = scorer.score(descriptions[SINGLE_GARMENT_ID])
    assert 0.0 <= score.overall <= 100.0


def test_an_outfit_without_shoes_reports_missing_footwear(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    description = descriptions[NO_FOOTWEAR_ID]
    issues = [
        issue
        for issue in scorer.score(description).issues
        if issue.code is IssueCode.missing_footwear
    ]

    assert len(issues) == 1
    assert issues[0].garment_refs
    assert set(issues[0].garment_refs) <= set(description.refs())


def test_an_all_neutral_outfit_keeps_a_defined_colour_subscore(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    harmony = scorer.score(descriptions[ALL_NEUTRAL_ID]).subscores.color_harmony
    assert math.isfinite(harmony)
    assert 0.0 <= harmony <= 100.0


def test_four_pieces_of_jewellery_overload_the_accessories(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    description = descriptions[FOUR_JEWELLERY_ID]
    issues = [
        issue
        for issue in scorer.score(description).issues
        if issue.code is IssueCode.accessory_overload
    ]

    assert len(issues) == 1
    jewellery = [
        garment.ref
        for garment in description.garments
        if garment.slot is GarmentSlot.jewelry
    ]
    assert len(jewellery) == 4
    assert set(jewellery) <= set(issues[0].garment_refs)


def test_every_accessory_is_named_even_beyond_the_reference_cap(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    settings = rule_settings()
    description = descriptions[FOUR_JEWELLERY_ID]
    accessories = [
        item.ref
        for item in description.garments
        if item.slot in settings.accessory_slots
    ]
    issues = [
        issue
        for issue in scorer.score(description).issues
        if issue.code is IssueCode.accessory_overload
    ]

    assert len(accessories) == 7
    assert len(accessories) > settings.max_issue_refs
    assert issues[0].garment_refs == accessories


def test_a_description_without_evidence_reports_one_informational_issue(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    description = descriptions[ALL_NONE_ID]
    score = scorer.score(description)

    assert len(score.issues) == 1
    assert score.issues[0].code is IssueCode.other
    assert score.issues[0].severity is IssueSeverity.info
    assert set(score.issues[0].garment_refs) <= set(description.refs())
    assert score.overall == score.subscores.color_harmony


def test_a_faultless_outfit_reports_nothing_more_than_information(
    descriptions: dict[str, OutfitDescription], scorer: RuleScorer
) -> None:
    score = scorer.score(descriptions[FAULTLESS_ID])
    assert all(issue.severity is IssueSeverity.info for issue in score.issues)


def test_the_worst_issues_lead_with_the_most_severe_one() -> None:
    score = OutfitScore(
        image_id="synthetic",
        overall=50.0,
        subscores=SubScores(
            color_harmony=50.0,
            formality_consistency=50.0,
            seasonality=50.0,
            proportion=50.0,
        ),
        issues=[
            Issue(code=IssueCode.other, severity=IssueSeverity.info, message="Information."),
            Issue(code=IssueCode.low_contrast, severity=IssueSeverity.minor, message="Minor."),
            Issue(code=IssueCode.hue_clash, severity=IssueSeverity.major, message="Major."),
        ],
        provenance=Provenance.fixture,
        source_model="test",
    )

    ranked = score.worst_issues(3)
    assert [issue.severity for issue in ranked] == [
        IssueSeverity.major,
        IssueSeverity.minor,
        IssueSeverity.info,
    ]


def test_category_keys_normalise_case_hyphens_and_spacing() -> None:
    assert category_key("  Cargo-Trousers  ") == "cargo trousers"
    assert category_key("T  SHIRT") == "t shirt"


def test_an_unmapped_category_gets_no_guessed_formality_level(scorer: RuleScorer) -> None:
    table = rule_settings().formality
    assert formality_level("unknown", table=table) is None
    assert formality_level("quantum poncho", table=table) is None

    description = described(
        garment(GarmentSlot.upper, "quantum poncho", area_fraction=0.4),
        garment(GarmentSlot.lower, "unknown", area_fraction=0.4),
        garment(GarmentSlot.footwear, "levitation boots", area_fraction=0.2),
    )
    score = scorer.score(description)

    assert IssueCode.formality_mismatch not in codes(score)
    assert score.subscores.formality_consistency == 100.0


@pytest.mark.parametrize(
    "weights",
    [
        {
            "color_harmony": -1.0,
            "formality_consistency": 1.0,
            "seasonality": 1.0,
            "proportion": 1.0,
        },
        {
            "color_harmony": float("nan"),
            "formality_consistency": 1.0,
            "seasonality": 1.0,
            "proportion": 1.0,
        },
        {
            "color_harmony": 0.0,
            "formality_consistency": 0.0,
            "seasonality": 0.0,
            "proportion": 0.0,
        },
        {"color_harmony": 1.0, "formality_consistency": 1.0, "seasonality": 1.0},
    ],
)
def test_unusable_weights_are_rejected(weights: dict[str, float]) -> None:
    config = copy.deepcopy(load_scoring_config())
    config["weights"] = weights

    with pytest.raises(ConfigError):
        rule_settings(config)


def test_the_shipped_configuration_is_accepted_verbatim() -> None:
    settings = rule_settings()
    assert set(settings.weights) == set(SubScores.model_fields)
    assert settings.base == 100.0


@pytest.mark.parametrize(
    "section",
    [
        "color",
        "formality",
        "season",
        "harmony",
        "proportion",
        "weights",
        "penalties",
        "defaults",
    ],
)
def test_a_missing_scoring_section_is_rejected(section: str) -> None:
    config = copy.deepcopy(load_scoring_config())
    del config[section]

    with pytest.raises(ConfigError, match=f"no {section} section"):
        rule_settings(config)


@pytest.mark.parametrize(
    ("section", "overrides", "match"),
    [
        ("formality", {"levels": {"Cargo-Trousers": 1}}, "is not normalised"),
        ("formality", {"max_spread": 3, "major_spread": 2}, "major_spread must not be below"),
        ("season", {"fabric_warmth": {"na": 0.5}}, "must not score the na member"),
        ("season", {"sleeve_exposure": {"diagonal": 0.5}}, "is not a SleeveLength value"),
        ("proportion", {"core_slots": ["torso"]}, "is not a garment slot"),
    ],
)
def test_an_invalid_rule_table_entry_is_rejected(
    section: str, overrides: dict[str, object], match: str
) -> None:
    config = copy.deepcopy(load_scoring_config())
    config[section] = {**config[section], **overrides}

    with pytest.raises(ConfigError, match=match):
        rule_settings(config)


def test_a_missing_penalty_key_is_rejected() -> None:
    config = copy.deepcopy(load_scoring_config())
    del config["penalties"]["dominant_area"]

    with pytest.raises(ConfigError, match="penalties is missing dominant_area"):
        rule_settings(config)


def test_lowering_the_pattern_limit_adds_only_a_pattern_clash(
    scoring_config, descriptions: dict[str, OutfitDescription]
) -> None:
    description = descriptions[TWO_PATTERNS_ID]

    scoring_config(harmony={"max_patterns": 2})
    relaxed = RuleScorer().score(description)

    scoring_config(harmony={"max_patterns": 1})
    tightened = RuleScorer().score(description)

    assert IssueCode.pattern_clash not in codes(relaxed)
    assert codes(tightened) - codes(relaxed) == {IssueCode.pattern_clash}
    assert tightened.subscores.color_harmony < relaxed.subscores.color_harmony
    for axis in ("formality_consistency", "seasonality", "proportion"):
        assert getattr(tightened.subscores, axis) == getattr(relaxed.subscores, axis)


def test_a_plain_garment_beside_undescribed_ones_is_not_a_pattern_clash(
    scorer: RuleScorer,
) -> None:
    description = described(
        garment(GarmentSlot.upper, "t shirt", pattern=Pattern.pure_color),
        garment(GarmentSlot.lower, "jeans", pattern=Pattern.na),
        garment(GarmentSlot.footwear, "sneakers", pattern=Pattern.na),
    )
    assert IssueCode.pattern_clash not in codes(scorer.score(description))


def test_one_hue_family_with_a_flat_lightness_reads_as_monochrome(
    scorer: RuleScorer,
) -> None:
    description = described(
        garment(
            GarmentSlot.upper,
            "t shirt",
            color=ColorName.red,
            color_lab=lab_at(30.0, 50.0),
        ),
        garment(
            GarmentSlot.lower,
            "jeans",
            color=ColorName.red,
            color_lab=lab_at(32.0, 53.0),
        ),
        garment(
            GarmentSlot.footwear,
            "sneakers",
            color=ColorName.red,
            color_lab=lab_at(28.0, 51.0),
        ),
    )
    found = codes(scorer.score(description))

    assert IssueCode.monochrome_flat in found
    assert IssueCode.low_contrast not in found


def test_a_rainbow_that_chains_around_the_wheel_is_not_monochrome(
    scorer: RuleScorer,
) -> None:
    description = described(
        *(
            garment(
                GarmentSlot.jewelry,
                "stacked bangles",
                color=ColorName.red,
                color_lab=lab_at(index * 360.0 / 13, 50.0),
            )
            for index in range(13)
        )
    )
    found = codes(scorer.score(description))

    assert IssueCode.monochrome_flat not in found
    assert IssueCode.low_contrast in found


def test_the_colour_family_issue_names_only_the_hue_carrying_garments(
    scorer: RuleScorer,
) -> None:
    description = described(
        garment(GarmentSlot.headwear, "beanie", color=ColorName.black),
        garment(GarmentSlot.neckwear, "wool scarf", color=ColorName.white),
        garment(GarmentSlot.eyewear, "sunglasses", color=ColorName.unknown),
        garment(GarmentSlot.bag, "tote bag", color=ColorName.gray),
        garment(GarmentSlot.belt, "woven belt", color=ColorName.beige),
        garment(GarmentSlot.upper, "t shirt", color=ColorName.red, color_lab=lab_at(0.0, 50.0)),
        garment(GarmentSlot.lower, "jeans", color=ColorName.red, color_lab=lab_at(70.0, 50.0)),
        garment(
            GarmentSlot.footwear, "sneakers", color=ColorName.red, color_lab=lab_at(140.0, 50.0)
        ),
        garment(GarmentSlot.outer, "parka", color=ColorName.red, color_lab=lab_at(210.0, 50.0)),
    )
    issues = [
        issue
        for issue in scorer.score(description).issues
        if issue.code is IssueCode.too_many_colors
    ]

    assert len(issues) == 1
    assert issues[0].garment_refs == [item.ref for item in description.garments[5:]]


def test_a_dominant_dress_is_expected_but_a_dominant_top_is_not(
    scorer: RuleScorer,
) -> None:
    footwear = garment(GarmentSlot.footwear, "loafers", area_fraction=0.1)
    bag = garment(GarmentSlot.bag, "tote bag", area_fraction=0.1)

    dress = described(
        garment(GarmentSlot.dress, "wrap dress", area_fraction=0.8),
        footwear.model_copy(deep=True),
        bag.model_copy(deep=True),
    )
    top = described(
        garment(GarmentSlot.upper, "sweater", area_fraction=0.8),
        footwear.model_copy(deep=True),
        bag.model_copy(deep=True),
    )

    assert IssueCode.proportion_imbalance not in codes(scorer.score(dress))
    assert IssueCode.proportion_imbalance in codes(scorer.score(top))


def test_an_upper_that_dwarfs_the_lower_is_flagged(scorer: RuleScorer) -> None:
    description = described(
        garment(GarmentSlot.upper, "sweater", area_fraction=0.55),
        garment(GarmentSlot.lower, "shorts", area_fraction=0.15),
        garment(GarmentSlot.footwear, "sneakers", area_fraction=0.3),
    )
    issues = [
        issue
        for issue in scorer.score(description).issues
        if issue.code is IssueCode.proportion_imbalance
    ]

    assert len(issues) == 1
    assert set(issues[0].garment_refs) == {"upper_0", "lower_0"}


def test_an_outfit_tripping_both_area_rules_is_docked_by_both_penalties(
    scorer: RuleScorer,
) -> None:
    description = described(
        garment(GarmentSlot.upper, "sweater", area_fraction=0.7),
        garment(GarmentSlot.lower, "shorts", area_fraction=0.1),
        garment(GarmentSlot.footwear, "sneakers", area_fraction=0.2),
    )
    score = scorer.score(description)
    penalties = rule_settings().penalties
    issues = [
        issue for issue in score.issues if issue.code is IssueCode.proportion_imbalance
    ]

    assert len(issues) == 2
    assert score.subscores.proportion == pytest.approx(
        100.0 - penalties["dominant_area"] - penalties["upper_lower_ratio"]
    )


def test_a_heavy_fabric_beside_an_exposed_cut_adds_only_a_season_mismatch(
    scorer: RuleScorer,
) -> None:
    def outfit(fabric: Fabric) -> OutfitDescription:
        return described(
            garment(GarmentSlot.outer, "parka", fabric=fabric),
            garment(
                GarmentSlot.lower,
                "shorts",
                fabric=Fabric.cotton,
                length=LowerLength.three_point,
            ),
        )

    mild = scorer.score(outfit(Fabric.cotton))
    heavy = scorer.score(outfit(Fabric.furry))
    issues = [issue for issue in heavy.issues if issue.code is IssueCode.season_mismatch]

    assert codes(heavy) - codes(mild) == {IssueCode.season_mismatch}
    assert len(issues) == 1
    assert issues[0].severity is IssueSeverity.major
    assert heavy.subscores.seasonality < mild.subscores.seasonality
    for axis in ("color_harmony", "formality_consistency", "proportion"):
        assert getattr(heavy.subscores, axis) == getattr(mild.subscores, axis)


def test_the_overall_score_follows_the_configured_weights(
    scoring_config, descriptions: dict[str, OutfitDescription]
) -> None:
    weights = {
        "color_harmony": 7.0,
        "formality_consistency": 1.0,
        "seasonality": 1.0,
        "proportion": 1.0,
    }
    scoring_config(weights=weights)
    score = RuleScorer().score(descriptions[TWO_PATTERNS_ID])
    parts = score.subscores.model_dump()

    weighted = sum(weights[name] * parts[name] for name in weights) / sum(weights.values())
    plain = sum(parts.values()) / len(parts)

    assert score.overall == pytest.approx(weighted)
    assert score.overall != pytest.approx(plain)


def test_largest_measured_ranks_by_area_then_ref_and_ignores_confidence() -> None:
    smaller = garment(GarmentSlot.upper, ref="upper_9", area_fraction=0.2, confidence=1.0)
    tied_late = garment(GarmentSlot.upper, ref="upper_2", area_fraction=0.5, confidence=1.0)
    tied_early = garment(GarmentSlot.upper, ref="upper_1", area_fraction=0.5, confidence=0.1)
    unmeasured = garment(GarmentSlot.upper, ref="upper_0", confidence=1.0)
    measured_zero = garment(GarmentSlot.upper, ref="upper_8", area_fraction=0.0)

    assert largest_measured([smaller, tied_late, tied_early, unmeasured]) is tied_early
    assert largest_measured([unmeasured, measured_zero]) is measured_zero
    assert largest_measured([unmeasured]) is unmeasured
    assert largest_measured([]) is None


def test_every_rule_carries_a_message_template_whose_placeholders_match_its_emitter() -> None:
    for name, template in _MESSAGES.items():
        assert set(re.findall(r"{(\w+)", template)) == MESSAGE_KWARGS[name]


def test_the_scorer_holds_no_tuning_constant_of_its_own() -> None:
    source = Path(rules.__file__).read_text(encoding="utf-8")
    assert set(re.findall(r"(?<![\w.])\d+\.\d+", source)) <= {"0.0", "100.0"}
