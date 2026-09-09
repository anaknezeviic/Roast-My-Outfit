"""Ground-truth audit must expose ambiguities instead of silently choosing labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rmo.data.ground_truth_audit import GroundTruthAuditError, audit_ground_truth


DEFAULTS = {
    "upper_fabric": "na",
    "lower_fabric": "na",
    "outer_fabric": "na",
    "upper_pattern": "na",
    "lower_pattern": "na",
    "outer_pattern": "na",
    "has_shape": False,
}


def stage_mask(root: Path, image_id: str, label: int) -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), label, dtype=np.uint8)).save(
        directory / f"{image_id}_segm.png"
    )


def table(*records: tuple[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for image_id, overrides in records:
        rows.append({"image_id": image_id, **DEFAULTS, **overrides})
    return pd.DataFrame(rows).set_index("image_id", drop=False)


@pytest.fixture()
def staged(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_mask_presence_counts_a_regional_garment_even_when_texture_is_na(staged) -> None:
    stage_mask(staged, "upper", 1)
    report = audit_ground_truth(["upper"], table=table(("upper", {})))
    assert report.slot_presence["upper"] == 1
    assert report.regional_texture_both_na["upper"] == 1


def test_dress_reports_matching_and_conflicting_upper_lower_labels(staged) -> None:
    stage_mask(staged, "same", 4)
    stage_mask(staged, "conflict", 4)
    frame = table(
        (
            "same",
            {
                "upper_fabric": "chiffon",
                "lower_fabric": "chiffon",
                "upper_pattern": "floral",
                "lower_pattern": "floral",
                "has_shape": True,
            },
        ),
        (
            "conflict",
            {
                "upper_fabric": "cotton",
                "lower_fabric": "denim",
                "upper_pattern": "striped",
                "lower_pattern": "pure_color",
            },
        ),
    )
    report = audit_ground_truth(["same", "conflict"], table=frame)
    assert report.dress.images == 2
    assert report.dress.shape_annotated == 1
    assert report.dress.fabric.same == 1
    assert report.dress.fabric.conflict == 1
    assert report.dress.pattern.same == 1
    assert report.dress.pattern.conflict == 1


def test_romper_reports_one_sided_and_missing_regional_labels(staged) -> None:
    for image_id in ("upper-only", "lower-only", "missing"):
        stage_mask(staged, image_id, 21)
    frame = table(
        ("upper-only", {"upper_fabric": "cotton", "upper_pattern": "striped"}),
        ("lower-only", {"lower_fabric": "denim", "lower_pattern": "floral"}),
        ("missing", {}),
    )
    report = audit_ground_truth(frame.index, table=frame)
    assert report.romper.fabric.upper_only == 1
    assert report.romper.fabric.lower_only == 1
    assert report.romper.fabric.both_na == 1
    assert report.romper.pattern.upper_only == 1
    assert report.romper.pattern.lower_only == 1
    assert report.romper.pattern.both_na == 1


def test_audit_fails_when_a_requested_mask_is_missing(staged) -> None:
    with pytest.raises(GroundTruthAuditError, match="No parsing mask"):
        audit_ground_truth(["missing"], table=table(("missing", {})))


def test_audit_fails_on_unknown_or_repeated_ids(staged) -> None:
    stage_mask(staged, "known", 1)
    frame = table(("known", {}))
    with pytest.raises(GroundTruthAuditError, match="absent from the outfit table"):
        audit_ground_truth(["unknown"], table=frame)
    with pytest.raises(GroundTruthAuditError, match="repeats"):
        audit_ground_truth(["known", "known"], table=frame)


def test_audit_rejects_a_mask_without_a_configured_garment_slot(staged) -> None:
    stage_mask(staged, "skin-only", 13)
    with pytest.raises(GroundTruthAuditError, match="no configured garment slot"):
        audit_ground_truth(["skin-only"], table=table(("skin-only", {})))


def test_audit_rejects_an_incomplete_table(staged) -> None:
    stage_mask(staged, "known", 1)
    frame = pd.DataFrame({"image_id": ["known"], "has_shape": [False]}).set_index(
        "image_id", drop=False
    )
    with pytest.raises(GroundTruthAuditError, match="missing required audit columns"):
        audit_ground_truth(["known"], table=frame)
