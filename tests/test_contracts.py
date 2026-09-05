"""Cover the stage registry and the end-to-end pipeline contract."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from itertools import chain

import pytest
from PIL import Image

from conftest import EXPECTED_IDS, SAMPLE_ID
from rmo import pipeline
from rmo.imaging import ImageInput
from rmo.paths import repo_root
from rmo.perception.base import PerceptionModel
from rmo.perception.cnn_adapter import CNNPerception
from rmo.perception.dummy import DummyPerception
from rmo.perception.vlm import REGISTRY_NAME
from rmo.roast.dummy import DummyRoaster
from rmo.roast.gemini import GeminiRoaster
from rmo.roast.rules import RuleBasedRoaster
from rmo.schemas import OutfitDescription, OutfitScore, Provenance, RoastOutput
from rmo.scoring.dummy import DummyScorer
from rmo.scoring.rules import RuleScorer

REQUIRED_NAMES = [
    DummyPerception.name,
    DummyScorer.name,
    DummyRoaster.name,
    RuleBasedRoaster.name,
    GeminiRoaster.name,
    CNNPerception.name,
    RuleScorer.name,
    REGISTRY_NAME,
]

DEFAULT_NAMES = [
    DummyPerception.name,
    DummyScorer.name,
    DummyRoaster.name,
]

FALLBACK_LOG = ("rmo.pipeline", "INFO")

IMPORT_PROBE = (
    "import {module}; "
    "from rmo.pipeline import registered_names; "
    "print(*registered_names())"
)


class ReplayPerception(PerceptionModel):
    """Hand back a committed description without reading an image or a checkpoint."""

    name = "replay_perception"

    def __init__(self, records: dict[str, OutfitDescription]) -> None:
        self._records = {record.image_path: record for record in records.values()}

    def predict(self, image: ImageInput) -> OutfitDescription:
        return self._records[str(image)].model_copy(deep=True)


@pytest.fixture(scope="module")
def outfit_roaster() -> pipeline.OutfitRoaster:
    return pipeline.OutfitRoaster()


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_every_fixture_runs_end_to_end_with_grounded_references(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    outfit_roaster: pipeline.OutfitRoaster,
) -> None:
    description, score, roast = outfit_roaster.run(
        descriptions[image_id].image_path
    )

    assert [description.image_id, score.image_id, roast.image_id] == [image_id] * 3

    assert [
        description.provenance,
        score.provenance,
        roast.provenance,
    ] == [Provenance.fixture] * 3

    assert RoastOutput.model_validate(
        roast.model_dump(mode="json")
    ) == roast

    refs = set(description.refs())

    assert set(
        chain.from_iterable(
            issue.garment_refs
            for issue in score.issues
        )
    ) <= refs

    assert set(roast.grounded_garments) <= refs


def test_run_hands_the_image_to_perception_unchanged(
    descriptions: dict[str, OutfitDescription],
    outfit_roaster: pipeline.OutfitRoaster,
) -> None:
    with Image.open(
        repo_root() / descriptions[SAMPLE_ID].image_path
    ) as handle:
        description, _, _ = outfit_roaster.run(handle)

    assert description.image_id == SAMPLE_ID


def test_a_missing_stage_falls_back_to_the_registered_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="rmo.pipeline",
    ):
        built = pipeline.OutfitRoaster()

    assert [
        built.perception.name,
        built.scorer.name,
        built.roaster.name,
    ] == DEFAULT_NAMES

    assert [
        (record.name, record.levelname)
        for record in caplog.records
    ] == [FALLBACK_LOG] * 3


def test_a_supplied_stage_is_not_substituted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    perception = DummyPerception()

    with caplog.at_level(
        logging.INFO,
        logger="rmo.pipeline",
    ):
        built = pipeline.OutfitRoaster(perception)

    assert built.perception is perception

    assert [
        (record.name, record.levelname)
        for record in caplog.records
    ] == [FALLBACK_LOG] * 2


def test_the_registry_contains_the_expected_models() -> None:
    names = pipeline.registered_names()

    assert set(REQUIRED_NAMES) <= set(names)
    assert names == sorted(names)
    assert len(names) == len(set(names))

    defaults = [
        pipeline.DEFAULT_PERCEPTION,
        pipeline.DEFAULT_SCORER,
        pipeline.DEFAULT_ROASTER,
    ]

    assert defaults == DEFAULT_NAMES


def test_the_readme_stage_table_lists_every_registered_model() -> None:
    table = re.findall(
        r"^\| `--(?:perception|scorer|roaster)` \| (.+) \|$",
        (repo_root() / "README.md").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    documented = {name.strip(" `") for row in table for name in row.split(",")}
    assert documented == set(pipeline.registered_names())


def test_registering_a_taken_name_raises_without_shadowing() -> None:
    with pytest.raises(
        ValueError,
        match=pipeline.DEFAULT_PERCEPTION,
    ):
        pipeline.register(
            pipeline.DEFAULT_PERCEPTION,
            object,
        )

    assert isinstance(
        pipeline.create(pipeline.DEFAULT_PERCEPTION),
        DummyPerception,
    )


def test_creating_an_unregistered_name_raises() -> None:
    with pytest.raises(
        ValueError,
        match="no_such_model",
    ):
        pipeline.create("no_such_model")


def test_main_prints_one_schema_valid_roast(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline.main(
        [
            "--image",
            descriptions[SAMPLE_ID].image_path,
        ]
    ) == 0

    printed = capsys.readouterr().out.splitlines()

    assert len(printed) == 1

    assert RoastOutput.model_validate_json(
        printed[0]
    ).image_id == SAMPLE_ID


def test_the_verbose_flag_lowers_the_root_logger_to_debug(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    original = root.level
    try:
        root.setLevel(logging.INFO)
        assert pipeline.main(["--image", descriptions[SAMPLE_ID].image_path]) == 0
        assert root.level == logging.INFO
        assert (
            pipeline.main(["--image", descriptions[SAMPLE_ID].image_path, "--verbose"]) == 0
        )
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original)
    capsys.readouterr()


def test_the_json_flag_adds_the_description_and_the_score(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline.main(
        [
            "--image",
            descriptions[SAMPLE_ID].image_path,
            "--json",
        ]
    ) == 0

    document = json.loads(
        capsys.readouterr().out
    )

    assert sorted(document) == [
        "description",
        "roast",
        "score",
    ]

    assert OutfitDescription.model_validate(
        document["description"]
    ).image_id == SAMPLE_ID

    assert OutfitScore.model_validate(
        document["score"]
    ).image_id == SAMPLE_ID

    assert RoastOutput.model_validate(
        document["roast"]
    ).image_id == SAMPLE_ID


def test_main_accepts_an_explicit_roaster_name(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline.main(
        [
            "--image",
            descriptions[SAMPLE_ID].image_path,
            "--roaster",
            RuleBasedRoaster.name,
        ]
    ) == 0

    roast = RoastOutput.model_validate_json(
        capsys.readouterr().out
    )

    assert roast.source_model == RuleBasedRoaster.name


def test_the_module_entry_point_prints_one_schema_valid_roast(
    descriptions: dict[str, OutfitDescription],
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rmo.pipeline",
            "--image",
            descriptions[SAMPLE_ID].image_path,
        ],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert RoastOutput.model_validate_json(
        completed.stdout
    ).image_id == SAMPLE_ID


@pytest.mark.parametrize(
    "module",
    [
        "rmo.pipeline",
        "rmo.perception",
        "rmo.scoring",
    ],
)
def test_the_registry_fills_under_any_import_order(
    module: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            IMPORT_PROBE.format(module=module),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    names = completed.stdout.split()

    assert set(REQUIRED_NAMES) <= set(names)

    assert names == sorted(names)


@pytest.fixture(scope="module")
def rule_roaster(
    descriptions: dict[str, OutfitDescription],
) -> pipeline.OutfitRoaster:
    return pipeline.OutfitRoaster(
        ReplayPerception(descriptions),
        RuleScorer(),
        RuleBasedRoaster(),
    )


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_the_rule_scorer_and_the_rule_roaster_run_every_fixture_end_to_end(
    image_id: str,
    descriptions: dict[str, OutfitDescription],
    rule_roaster: pipeline.OutfitRoaster,
) -> None:
    description, score, roast = rule_roaster.run(
        descriptions[image_id].image_path
    )

    assert [description.image_id, score.image_id, roast.image_id] == [image_id] * 3

    assert score.source_model == RuleScorer.name

    assert roast.source_model == RuleBasedRoaster.name

    assert [score.provenance, roast.provenance] == [description.provenance] * 2

    refs = set(description.refs())

    assert set(
        chain.from_iterable(
            issue.garment_refs
            for issue in score.issues
        )
    ) <= refs

    assert set(roast.grounded_garments) <= refs


def test_main_accepts_an_explicit_scorer_name(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline.main(
        [
            "--image",
            descriptions[SAMPLE_ID].image_path,
            "--scorer",
            RuleScorer.name,
            "--json",
        ]
    ) == 0

    document = json.loads(
        capsys.readouterr().out
    )

    assert OutfitScore.model_validate(
        document["score"]
    ).source_model == RuleScorer.name


def test_main_accepts_an_explicit_perception_name(
    descriptions: dict[str, OutfitDescription],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pipeline.main(
        [
            "--image",
            descriptions[SAMPLE_ID].image_path,
            "--perception",
            DummyPerception.name,
            "--json",
        ]
    ) == 0

    document = json.loads(
        capsys.readouterr().out
    )

    assert OutfitDescription.model_validate(
        document["description"]
    ).image_id == SAMPLE_ID


def test_explicitly_selected_stages_are_never_replaced_by_a_fallback(
    descriptions: dict[str, OutfitDescription],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="rmo.pipeline",
    ):
        assert pipeline.main(
            [
                "--image",
                descriptions[SAMPLE_ID].image_path,
                "--perception",
                DummyPerception.name,
                "--scorer",
                RuleScorer.name,
                "--roaster",
                RuleBasedRoaster.name,
            ]
        ) == 0

    capsys.readouterr()

    assert [
        (record.name, record.levelname)
        for record in caplog.records
    ] == []


@pytest.mark.parametrize(
    "flag",
    [
        "--perception",
        "--scorer",
        "--roaster",
    ],
)
def test_an_unknown_stage_name_raises(
    descriptions: dict[str, OutfitDescription],
    flag: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="no_such_model",
    ):
        pipeline.main(
            [
                "--image",
                descriptions[SAMPLE_ID].image_path,
                flag,
                "no_such_model",
            ]
        )
