"""The SmolVLM preflight must fail closed on bad staged data or bad targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rmo import paths
from rmo.data.smolvlm_preflight import (
    DatasetPreflightError,
    audit_splits,
    verify_split_manifest,
)
from rmo.splits import group_key_for

IMAGE_ID = "WOMEN-Tees_Tanks-id_00000001-01_7_additional"
VAL_ID = "WOMEN-Tees_Tanks-id_00000002-01_7_additional"
TEST_ID = "WOMEN-Tees_Tanks-id_00000003-01_7_additional"


def _row(image_id: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "is_full_body": True,
        "upper_fabric": "cotton",
        "upper_pattern": "striped",
        "outer_fabric": "na",
        "outer_pattern": "na",
        "lower_fabric": "na",
        "lower_pattern": "na",
        "sleeve_length": "short",
        "lower_length": "na",
        "neckline": "round",
        "has_shape": True,
        "caption": "",
        "category_from_filename": "Tees_Tanks",
    }


def _stage_pair(
    root: Path,
    image_id: str,
    *,
    mask_shape: tuple[int, int] = (8, 8),
    rgb: tuple[int, int, int] = (0, 0, 255),
) -> None:
    images = root / "raw" / "images"
    parsing = root / "raw" / "parsing"
    images.mkdir(parents=True, exist_ok=True)
    parsing.mkdir(parents=True, exist_ok=True)
    pixels = np.empty((8, 8, 3), dtype=np.uint8)
    pixels[...] = rgb
    Image.fromarray(pixels).save(images / f"{image_id}.jpg", quality=100)
    mask = np.ones(mask_shape, dtype=np.uint8)
    Image.fromarray(mask).save(parsing / f"{image_id}_segm.png")


def _stage_splits(root: Path, *, duplicate_train: bool = False) -> None:
    directory = root / "processed" / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    (root / "processed" / "outfits.parquet").write_bytes(b"synthetic parquet placeholder")
    ids = {"train": [IMAGE_ID], "val": [VAL_ID], "test": [TEST_ID]}
    if duplicate_train:
        ids["train"].append(IMAGE_ID)
    for name, values in ids.items():
        (directory / f"{name}.txt").write_text(
            "".join(f"{value}\n" for value in values), encoding="utf-8"
        )
    manifest = {
        "counts": {name: len(values) for name, values in ids.items()},
        "n_groups": len({group_key_for(value) for values in ids.values() for value in values}),
        "sha256": {
            f"{name}.txt": paths.file_sha256(directory / f"{name}.txt")
            for name in ids
        },
        "split_seed": 20260101,
    }
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def _table() -> pd.DataFrame:
    frame = pd.DataFrame([_row(IMAGE_ID), _row(VAL_ID), _row(TEST_ID)])
    return frame.set_index("image_id", drop=False)


def test_manifest_rejects_duplicate_split_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path, duplicate_train=True)
    with pytest.raises(DatasetPreflightError, match="duplicate"):
        verify_split_manifest()


def test_manifest_rejects_hash_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    manifest_path = tmp_path / "processed" / "splits" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["train.txt"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetPreflightError, match="SHA-256"):
        verify_split_manifest()


def test_audit_accepts_complete_train_and_val(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    for image_id in (IMAGE_ID, VAL_ID, TEST_ID):
        _stage_pair(tmp_path, image_id)
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: _table())

    report = audit_splits(("train", "val"), progress_every=0)

    assert report.ok
    assert report.splits["train"].built == 1
    assert report.splits["val"].built == 1
    assert report.splits["train"].slot_counts == {"upper": 1}
    assert report.splits["train"].field_values["fabric"] == {"cotton": 1}
    assert report.splits["train"].field_values["pattern"] == {"striped": 1}


def test_audit_rejects_mask_dimension_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    _stage_pair(tmp_path, IMAGE_ID, mask_shape=(7, 8))
    _stage_pair(tmp_path, VAL_ID)
    _stage_pair(tmp_path, TEST_ID)
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: _table())

    report = audit_splits(("train",), progress_every=0)

    assert not report.ok
    assert report.splits["train"].built == 0
    assert report.splits["train"].errors["dimension_mismatch"] == 1


def test_audit_rejects_corrupt_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    _stage_pair(tmp_path, IMAGE_ID)
    _stage_pair(tmp_path, VAL_ID)
    _stage_pair(tmp_path, TEST_ID)
    (tmp_path / "raw" / "images" / f"{IMAGE_ID}.jpg").write_bytes(b"not an image")
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: _table())

    report = audit_splits(("train",), progress_every=0)

    assert not report.ok
    assert report.splits["train"].errors["image_decode"] == 1


def test_validation_value_absent_from_train_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    for image_id in (IMAGE_ID, VAL_ID, TEST_ID):
        _stage_pair(tmp_path, image_id)
    frame = _table()
    frame.loc[VAL_ID, "upper_fabric"] = "leather"
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: frame)

    report = audit_splits(("train", "val"), progress_every=0)

    assert not report.ok
    assert report.coverage_errors == ["val has fabric value(s) absent from train: leather"]



def test_validation_color_absent_from_train_is_warning_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    _stage_pair(tmp_path, IMAGE_ID)
    _stage_pair(tmp_path, VAL_ID, rgb=(128, 0, 255))
    _stage_pair(tmp_path, TEST_ID)
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: _table())

    report = audit_splits(("train", "val"), progress_every=0)

    assert report.ok
    assert report.coverage_errors == []
    assert report.coverage_warnings == [
        "val has derived color value(s) absent from train: violet=1"
    ]

def test_audit_rejects_mask_without_configured_garments(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    _stage_pair(tmp_path, IMAGE_ID)
    _stage_pair(tmp_path, VAL_ID)
    _stage_pair(tmp_path, TEST_ID)
    # class 13 is not mapped to any RMO garment slot in perception.yaml
    Image.fromarray(np.full((8, 8), 13, dtype=np.uint8)).save(
        tmp_path / "raw" / "parsing" / f"{IMAGE_ID}_segm.png"
    )
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: _table())

    report = audit_splits(("train",), progress_every=0)

    assert not report.ok
    assert report.splits["train"].errors["mask_no_garments"] == 1


def test_audit_rejects_noncanonical_annotation_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    _stage_splits(tmp_path)
    for image_id in (IMAGE_ID, VAL_ID, TEST_ID):
        _stage_pair(tmp_path, image_id)
    frame = _table()
    frame.loc[IMAGE_ID, "upper_fabric"] = "silk"
    monkeypatch.setattr("rmo.data.smolvlm_preflight.load_outfit_table", lambda path=None: frame)

    report = audit_splits(("train",), progress_every=0)

    assert not report.ok
    assert report.splits["train"].errors["annotation_vocabulary"] == 1
