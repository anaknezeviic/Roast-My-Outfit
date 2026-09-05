"""Check that staged-input reporting sees exactly what is on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rmo.data.preflight import (
    InputReport,
    PreflightError,
    photo_path,
    report_for_ids,
    require,
)

SPLIT_IDS = ("alpha", "beta", "gamma")


def stage_photo(root: Path, image_id: str) -> None:
    directory = root / "raw" / "images"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(directory / f"{image_id}.jpg")


def stage_mask(root: Path, image_id: str, *, suffix: str = "_segm") -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(directory / f"{image_id}{suffix}.png")


@pytest.fixture()
def data_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_photo_path_follows_the_data_root(data_root) -> None:
    assert photo_path("one") == data_root / "raw" / "images" / "one.jpg"


def test_a_fully_staged_set_reports_nothing_missing(data_root) -> None:
    for image_id in SPLIT_IDS:
        stage_photo(data_root, image_id)
        stage_mask(data_root, image_id)
    report = report_for_ids(SPLIT_IDS, label="staged")
    assert report.ok
    assert report.total == 3
    assert report.missing_photos == ()
    assert report.missing_masks == ()


def test_a_missing_photograph_is_reported(data_root) -> None:
    for image_id in SPLIT_IDS:
        stage_mask(data_root, image_id)
    stage_photo(data_root, "alpha")
    report = report_for_ids(SPLIT_IDS)
    assert not report.ok
    assert report.missing_photos == ("beta", "gamma")
    assert report.missing_masks == ()


def test_a_missing_mask_is_reported(data_root) -> None:
    for image_id in SPLIT_IDS:
        stage_photo(data_root, image_id)
    stage_mask(data_root, "alpha")
    report = report_for_ids(SPLIT_IDS)
    assert report.missing_photos == ()
    assert report.missing_masks == ("beta", "gamma")


def test_a_plain_named_mask_counts_as_staged(data_root) -> None:
    stage_photo(data_root, "alpha")
    stage_mask(data_root, "alpha", suffix="")
    assert report_for_ids(["alpha"]).ok


def test_an_empty_set_is_trivially_complete(data_root) -> None:
    report = report_for_ids([])
    assert report.ok
    assert report.total == 0


def test_require_passes_a_complete_report(data_root) -> None:
    stage_photo(data_root, "alpha")
    stage_mask(data_root, "alpha")
    assert require(report_for_ids(["alpha"])) is None


def test_require_names_the_first_missing_inputs(data_root) -> None:
    report = report_for_ids(SPLIT_IDS, label="val")
    with pytest.raises(PreflightError, match="alpha, beta, gamma"):
        require(report)


def test_require_truncates_the_examples(data_root) -> None:
    report = report_for_ids(SPLIT_IDS, label="val")
    with pytest.raises(PreflightError) as excinfo:
        require(report, examples=1)
    assert "alpha" in str(excinfo.value)
    assert "beta" not in str(excinfo.value)


def test_summary_counts_both_kinds_of_shortfall(data_root) -> None:
    stage_photo(data_root, "alpha")
    report = report_for_ids(SPLIT_IDS, label="val")
    assert report.summary() == "val: 3 ids, 2 photographs and 3 parsing masks missing"


def test_the_report_is_immutable() -> None:
    report = InputReport(label="val", total=1, missing_photos=(), missing_masks=())
    with pytest.raises(AttributeError):
        report.total = 2
