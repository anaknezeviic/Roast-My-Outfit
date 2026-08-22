"""Cover the fixture-replay implementations of the three pipeline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import EXPECTED_IDS
from rmo import fixtures
from rmo.paths import repo_root
from rmo.perception.base import PerceptionModel
from rmo.perception.dummy import DummyPerception
from rmo.roast.base import RoastGenerator
from rmo.roast.dummy import DummyRoaster
from rmo.schemas import OutfitDescription, OutfitScore, Provenance, RoastOutput
from rmo.scoring.base import ScoringModel
from rmo.scoring.dummy import DummyScorer

SAMPLE_ID = "fixture_000"

UNKNOWN_ID = "not_a_fixture"

BLANK_PIXELS = np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def perception() -> DummyPerception:
    return DummyPerception()


@pytest.fixture(scope="module")
def scorer() -> DummyScorer:
    return DummyScorer()


@pytest.fixture(scope="module")
def roaster() -> DummyRoaster:
    return DummyRoaster()


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_the_dummy_chain_replays_the_corpus(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
    perception: DummyPerception,
    scorer: DummyScorer,
    roaster: DummyRoaster,
) -> None:
    description = perception.predict(descriptions[image_id].image_path)
    score = scorer.score(description)
    roast = roaster.generate(description, score)
    assert description == descriptions[image_id]
    assert score == scores[image_id]
    assert roast == roasts[image_id]
    assert score.provenance == description.provenance
    assert roast.provenance == description.provenance


def test_predict_accepts_a_path_object(
    descriptions: dict[str, OutfitDescription], perception: DummyPerception
) -> None:
    path = Path(descriptions[SAMPLE_ID].image_path)
    assert perception.predict(path).image_id == SAMPLE_ID


def test_predict_accepts_an_image_opened_from_a_path(
    descriptions: dict[str, OutfitDescription], perception: DummyPerception
) -> None:
    with Image.open(repo_root() / descriptions[SAMPLE_ID].image_path) as handle:
        assert perception.predict(handle).image_id == SAMPLE_ID


@pytest.mark.parametrize("image", [BLANK_PIXELS, Image.fromarray(BLANK_PIXELS)])
def test_predict_refuses_an_image_that_names_no_file(
    image: object, perception: DummyPerception
) -> None:
    with pytest.raises(TypeError, match="needs an image path"):
        perception.predict(image)


@pytest.mark.parametrize("provenance", [Provenance.gt, Provenance.predicted])
def test_provenance_is_taken_from_the_description_not_the_corpus(
    provenance: Provenance,
    descriptions: dict[str, OutfitDescription],
    scorer: DummyScorer,
    roaster: DummyRoaster,
) -> None:
    description = descriptions[SAMPLE_ID].model_copy(deep=True)
    description.provenance = provenance
    score = scorer.score(description)
    assert score.provenance is provenance
    assert roaster.generate(description, score).provenance is provenance


def test_an_unknown_image_id_is_refused_by_every_stage(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    perception: DummyPerception,
    scorer: DummyScorer,
    roaster: DummyRoaster,
) -> None:
    description = descriptions[SAMPLE_ID].model_copy(deep=True)
    description.image_id = UNKNOWN_ID
    with pytest.raises(KeyError, match=UNKNOWN_ID):
        perception.predict(f"{UNKNOWN_ID}.png")
    with pytest.raises(KeyError, match=UNKNOWN_ID):
        scorer.score(description)
    with pytest.raises(KeyError, match=UNKNOWN_ID):
        roaster.generate(description, scores[SAMPLE_ID])


def test_the_dummies_implement_the_stage_interfaces(
    perception: DummyPerception, scorer: DummyScorer, roaster: DummyRoaster
) -> None:
    assert isinstance(perception, PerceptionModel)
    assert isinstance(scorer, ScoringModel)
    assert isinstance(roaster, RoastGenerator)


def test_every_stage_carries_its_own_name(
    perception: DummyPerception, scorer: DummyScorer, roaster: DummyRoaster
) -> None:
    assert [perception.name, scorer.name, roaster.name] == [
        "dummy_perception",
        "dummy_scorer",
        "dummy_roaster",
    ]


def test_predict_batch_returns_one_description_per_image_in_order(
    descriptions: dict[str, OutfitDescription], perception: DummyPerception
) -> None:
    image_ids = [EXPECTED_IDS[3], EXPECTED_IDS[0], EXPECTED_IDS[3], EXPECTED_IDS[-1]]
    images = [descriptions[image_id].image_path for image_id in image_ids]
    assert [record.image_id for record in perception.predict_batch(images)] == image_ids


def test_results_are_deep_copies_that_do_not_disturb_the_next_call(
    descriptions: dict[str, OutfitDescription],
    scores: dict[str, OutfitScore],
    roasts: dict[str, RoastOutput],
    perception: DummyPerception,
    scorer: DummyScorer,
    roaster: DummyRoaster,
) -> None:
    image = descriptions[SAMPLE_ID].image_path
    description = perception.predict(image)
    score = scorer.score(description)
    roast = roaster.generate(description, score)

    description.garments[0].category = "mutated"
    score.subscores.color_harmony = 0.0
    roast.suggestions = ["mutated"]

    assert perception.predict(image) == descriptions[SAMPLE_ID]
    assert scorer.score(descriptions[SAMPLE_ID]) == scores[SAMPLE_ID]
    assert roaster.generate(descriptions[SAMPLE_ID], scores[SAMPLE_ID]) == roasts[SAMPLE_ID]


def test_each_stage_loads_its_corpus_once_per_instance(
    descriptions: dict[str, OutfitDescription], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original = fixtures.load_records

    def counting_load_records(name, model):
        calls.append(name)
        return original(name, model)

    monkeypatch.setattr(fixtures, "load_records", counting_load_records)

    perception = DummyPerception()
    scorer = DummyScorer()
    roaster = DummyRoaster()
    assert calls == [
        "outfit_descriptions.jsonl",
        "outfit_scores.jsonl",
        "roast_outputs.jsonl",
    ]

    description = perception.predict(descriptions[SAMPLE_ID].image_path)
    roaster.generate(description, scorer.score(description))
    assert len(calls) == 3

    DummyPerception()
    assert len(calls) == 4
