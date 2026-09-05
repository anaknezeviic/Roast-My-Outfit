"""Guard the committed fixture corpus and the generator that produces it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pytest

from conftest import EXPECTED_IDS
from rmo.imaging import load_image
from rmo.paths import fixtures_dir, repo_root
from rmo.schemas import (
    NEUTRAL_COLORS,
    SCHEMA_VERSION,
    ColorName,
    GarmentSlot,
    IssueCode,
    IssueSeverity,
    OutfitDescription,
    OutfitScore,
    Provenance,
    RoastOutput,
    Tone,
)
from rmo.scoring.palette import mean_lab, nearest_color_name

FLAGGED_CATEGORIES = [
    "body",
    "age",
    "race",
    "gender",
    "attractiveness",
    "disability",
    "profanity",
    "implicature",
]

NEGATIVE_CATEGORY = "garment_critique"

CANVAS_SIZE = (128, 256)

FOOTWEAR_WORDS = ("shoe", "boot", "sneaker", "trainer", "heel", "sandal", "loafer", "pump")


def _lines(name: str) -> list[str]:
    """Return the lines of one committed corpus file."""
    return (fixtures_dir() / name).read_text(encoding="utf-8").splitlines()


def _snapshot(root: Path) -> dict[str, bytes]:
    """Return every file below ``root``, keyed by its posix-style relative path."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def probes() -> list[dict[str, object]]:
    """Return the committed safety probes in file order."""
    return [json.loads(line) for line in _lines("safety_probes.jsonl")]


@pytest.mark.parametrize(
    "name", ["outfit_descriptions.jsonl", "outfit_scores.jsonl", "roast_outputs.jsonl"]
)
def test_every_corpus_file_holds_one_record_per_fixture(name: str) -> None:
    assert len(_lines(name)) == len(EXPECTED_IDS)


def test_the_three_corpora_share_one_id_sequence(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
) -> None:
    assert list(descriptions) == EXPECTED_IDS
    assert list(scores) == EXPECTED_IDS
    assert list(roasts) == EXPECTED_IDS


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_records_are_marked_as_fixture_data(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
) -> None:
    for record in (descriptions[image_id], scores[image_id], roasts[image_id]):
        assert record.provenance is Provenance.fixture
        assert record.source_model == "fixture"
        assert record.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_image_path_resolves_to_a_committed_png(
    image_id: str, descriptions: dict[str, OutfitDescription]
) -> None:
    record = descriptions[image_id]
    assert record.image_path == f"data/fixtures/images/{image_id}.png"
    assert load_image(record.image_path).size == CANVAS_SIZE


def test_the_image_directory_holds_only_the_corpus() -> None:
    stems = sorted(path.stem for path in (fixtures_dir() / "images").iterdir())
    assert stems == sorted(EXPECTED_IDS)


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_garment_colours_are_measured_from_their_own_band(
    image_id: str, descriptions: dict[str, OutfitDescription]
) -> None:
    for garment in descriptions[image_id].garments:
        if garment.color is ColorName.unknown:
            assert garment.color_lab is None
            assert garment.color_lab_source is None
            assert garment.area_fraction is None
            continue
        assert garment.color_lab is not None
        assert nearest_color_name(garment.color_lab) is garment.color
        assert garment.color_lab_source == "mask"
        assert garment.area_fraction is not None and garment.area_fraction > 0.0


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_garments_are_listed_in_slot_declaration_order(
    image_id: str, descriptions: dict[str, OutfitDescription]
) -> None:
    slots = list(GarmentSlot)
    positions = [slots.index(garment.slot) for garment in descriptions[image_id].garments]
    assert positions == sorted(positions)


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_area_fractions_cover_the_canvas(
    image_id: str, descriptions: dict[str, OutfitDescription]
) -> None:
    fractions = [garment.area_fraction for garment in descriptions[image_id].garments]
    if any(fraction is None for fraction in fractions):
        pytest.skip("this fixture carries no measured garments")
    assert sum(fractions) == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_issue_and_roast_references_name_real_garments(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
) -> None:
    refs = set(descriptions[image_id].refs())
    cited = chain.from_iterable(issue.garment_refs for issue in scores[image_id].issues)
    assert set(cited) <= refs
    assert set(roasts[image_id].grounded_garments) <= refs


def test_the_corpus_exercises_every_enumerated_value(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
) -> None:
    issues = [issue for score in scores.values() for issue in score.issues]
    assert {issue.code for issue in issues} == set(IssueCode)
    assert {issue.severity for issue in issues} == set(IssueSeverity)
    assert {roast.tone for roast in roasts.values()} == set(Tone)
    slots = {garment.slot for record in descriptions.values() for garment in record.garments}
    assert slots == set(GarmentSlot) - {GarmentSlot.other}


def test_single_garment_fixture_has_no_pair(descriptions: dict[str, OutfitDescription]) -> None:
    garments = descriptions["fx_deg_00"].garments
    assert len(garments) == 1
    assert garments[0].slot is GarmentSlot.upper


def test_footwear_free_fixture_flags_it_without_describing_shoes(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
) -> None:
    assert descriptions["fx_deg_01"].by_slot(GarmentSlot.footwear) == []
    codes = {issue.code for issue in scores["fx_deg_01"].issues}
    assert IssueCode.missing_footwear in codes
    described = f"{descriptions['fx_deg_01'].caption} {roasts['fx_deg_01'].roast}".lower()
    assert [word for word in FOOTWEAR_WORDS if word in described] == []


def test_dress_only_fixture_has_no_lower_garment(descriptions: dict[str, OutfitDescription]) -> None:
    record = descriptions["fx_deg_02"]
    assert record.by_slot(GarmentSlot.dress) != []
    assert record.by_slot(GarmentSlot.lower) == []


def test_all_neutral_fixture_is_achromatic(descriptions: dict[str, OutfitDescription]) -> None:
    for garment in descriptions["fx_deg_03"].garments:
        assert garment.color in NEUTRAL_COLORS
        assert garment.color_lab is not None
        _, green_red, blue_yellow = garment.color_lab
        assert abs(green_red) < 1.0 and abs(blue_yellow) < 1.0


def test_accessory_overload_fixture_numbers_four_jewellery_refs(
    descriptions: dict[str, OutfitDescription],
) -> None:
    record = descriptions["fx_deg_04"]
    assert len(record.garments) == 11
    assert len(record.by_slot(GarmentSlot.jewelry)) == 4
    assert "jewelry_3" in record.refs()


def test_unset_fixture_carries_no_attributes(descriptions: dict[str, OutfitDescription]) -> None:
    record = descriptions["fx_deg_05"]
    assert record.caption == ""
    for garment in record.garments:
        assert garment.color is ColorName.unknown
        assert garment.confidence == 0.0
        assert garment.sleeve_length is None
        assert garment.length is None
        assert garment.neckline is None


def test_duplicate_slot_fixture_is_separable_only_by_ref(
    descriptions: dict[str, OutfitDescription],
) -> None:
    record = descriptions["fx_deg_06"]
    upper = record.by_slot(GarmentSlot.upper)
    assert len(upper) == 2
    assert upper[0].category == upper[1].category
    assert [garment.ref for garment in upper] == ["upper_0", "upper_1"]


def test_long_caption_fixture_survives_the_utf8_round_trip(
    descriptions: dict[str, OutfitDescription],
) -> None:
    caption = descriptions["fx_deg_07"].caption
    assert len(caption) == 2000
    assert caption == caption.strip()
    assert not caption.isascii()


@pytest.mark.parametrize("image_id", ["fx_adv_00", "fx_adv_01"])
def test_adversarial_fixtures_report_only_major_issues(
    image_id: str, scores: dict[str, OutfitScore]
) -> None:
    issues = scores[image_id].issues
    assert len(issues) >= 3
    assert {issue.severity for issue in issues} == {IssueSeverity.major}


def test_tie_break_reorders_every_reported_issue(scores: dict[str, OutfitScore]) -> None:
    score = scores["fx_adv_01"]
    ranked = [(issue.code, tuple(issue.garment_refs)) for issue in score.worst_issues(3)]
    written = [(issue.code, tuple(issue.garment_refs)) for issue in score.issues[:3]]
    assert all(left != right for left, right in zip(ranked, written))


def test_the_faultless_outfit_is_the_only_one_complimented(
    scores: dict[str, OutfitScore], roasts: dict[str, RoastOutput]
) -> None:
    assert [image_id for image_id, score in scores.items() if not score.issues] == ["fx_adv_02"]
    complimented = [image_id for image_id, roast in roasts.items() if roast.tone is Tone.compliment]
    assert complimented == ["fx_adv_02"]


def test_probes_are_numbered_in_file_order(probes: list[dict[str, object]]) -> None:
    assert [probe["probe_id"] for probe in probes] == [f"sp_{index:02d}" for index in range(60)]


def test_probes_split_forty_must_catch_from_twenty_must_pass(
    probes: list[dict[str, object]],
) -> None:
    flagged = [probe for probe in probes if probe["must_flag"] is True]
    assert len(flagged) == 40
    assert len(probes) - len(flagged) == 20


def test_every_flagged_category_carries_at_least_five_probes(
    probes: list[dict[str, object]],
) -> None:
    counts = Counter(probe["category"] for probe in probes if probe["must_flag"])
    assert sorted(counts) == sorted(FLAGGED_CATEGORIES)
    assert min(counts.values()) >= 5


def test_probe_records_are_well_formed(probes: list[dict[str, object]]) -> None:
    for probe in probes:
        assert set(probe) == {"probe_id", "text", "category", "must_flag"}
        assert isinstance(probe["text"], str) and probe["text"].strip()
        assert probe["must_flag"] is (probe["category"] != NEGATIVE_CATEGORY)


def test_regenerating_the_corpus_reproduces_it_byte_for_byte(tmp_path: Path) -> None:
    script = repo_root() / "scripts" / "make_fixtures.py"
    environment = {**os.environ, "RMO_DATA_ROOT": str(tmp_path)}
    snapshots = []
    for _ in range(2):
        subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root(),
            env=environment,
            check=True,
            capture_output=True,
        )
        snapshots.append(_snapshot(tmp_path / "fixtures"))

    first, second = snapshots
    assert sorted(first) == sorted(second)
    assert [name for name in first if first[name] != second[name]] == []

    committed = _snapshot(fixtures_dir())
    second.pop("golden_v1.0.0.jsonl")
    committed.pop("golden_v1.0.0.jsonl")
    assert sorted(second) == sorted(committed)
    assert [name for name in second if second[name] != committed[name]] == []


def test_committed_colour_lab_is_the_whole_canvas_mean(
    descriptions: dict[str, OutfitDescription],
) -> None:
    image = load_image(fixtures_dir() / "images" / "fx_deg_00.png")
    measured = tuple(round(value, 2) for value in mean_lab(np.asarray(image, dtype=np.uint8)))
    garment = descriptions["fx_deg_00"].garments[0]
    assert garment.color_lab == measured
