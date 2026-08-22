"""Guard the golden corpus against removed, renamed or tightened fields."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import EXPECTED_IDS
from rmo.paths import fixtures_dir
from rmo.schemas import OutfitDescription

REMEDY = "restore the field or constraint that changed."


def _first_divergence(written: object, dumped: object, path: str = "") -> str | None:
    """Return the path of the first written value that the dump does not reproduce."""
    if isinstance(written, dict):
        if not isinstance(dumped, dict):
            return path
        for key, value in written.items():
            if key not in dumped:
                return f"{path}.{key}"
            divergence = _first_divergence(value, dumped[key], f"{path}.{key}")
            if divergence is not None:
                return divergence
        return None

    if isinstance(written, list):
        if not isinstance(dumped, list) or len(written) != len(dumped):
            return path
        for index, value in enumerate(written):
            divergence = _first_divergence(value, dumped[index], f"{path}[{index}]")
            if divergence is not None:
                return divergence
        return None

    if written == dumped and type(written) is type(dumped):
        return None
    return path


@pytest.fixture(scope="module")
def golden() -> dict[str, dict[str, object]]:
    """Return the golden records, keyed by image id."""
    path = fixtures_dir() / "golden_v1.0.0.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {record["image_id"]: record for record in records}


def test_the_golden_corpus_covers_every_fixture(golden: dict[str, dict[str, object]]) -> None:
    assert sorted(golden) == sorted(EXPECTED_IDS)


@pytest.mark.parametrize("image_id", EXPECTED_IDS)
def test_every_golden_record_still_parses_with_every_key_intact(
    image_id: str, golden: dict[str, dict[str, object]]
) -> None:
    record = golden[image_id]
    try:
        parsed = OutfitDescription.model_validate(record)
    except ValidationError as error:
        pytest.fail(f"Golden record {image_id!r} no longer parses; {REMEDY}\n{error}")

    divergence = _first_divergence(record, parsed.model_dump(mode="json"))
    assert divergence is None, (
        f"Golden record {image_id!r} no longer reproduces {divergence}; {REMEDY}"
    )


def test_a_key_added_anywhere_is_accepted() -> None:
    written = {"image_id": "a", "garments": [{"color": "red"}]}
    dumped = {"image_id": "a", "garments": [{"color": "red", "sheen": None}], "mood": "calm"}
    assert _first_divergence(written, dumped) is None


@pytest.mark.parametrize(
    ("dumped", "expected"),
    [
        ({"image_id": "a", "garments": [{"colour": "red"}]}, ".garments[0].color"),
        ({"image_id": "a", "garments": [{"color": "blue"}]}, ".garments[0].color"),
        ({"image_id": "b", "garments": [{"color": "red"}]}, ".image_id"),
        ({"image_id": "a", "garments": []}, ".garments"),
    ],
)
def test_a_removed_or_changed_value_is_reported_by_path(
    dumped: dict[str, object], expected: str
) -> None:
    written = {"image_id": "a", "garments": [{"color": "red"}]}
    assert _first_divergence(written, dumped) == expected
