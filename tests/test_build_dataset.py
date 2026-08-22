"""Outfit table tests using local staged fixtures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from rmo.data.build_dataset import DatasetError, build_dataset, build_outfit_table


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def staged_data(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    names = [
        "MEN-Denim-id_00000001-01_1_front",
        "WOMEN-Blouses_Shirts-id_00000002-02_2_side",
    ]
    shape_codes = "0 0 0 0 0 0 0 0 0 0 0 0"
    write(
        raw / "labels" / "shape_anno_all.txt",
        "".join(f"{name}.jpg {shape_codes}\n" for name in names),
    )
    write(
        raw / "labels" / "fabric_ann.txt",
        "".join(f"{name}.jpg 0 1 7\n" for name in names),
    )
    write(
        raw / "labels" / "pattern_ann.txt",
        "".join(f"{name}.jpg 3 2 7\n" for name in names),
    )
    write(
        raw / "captions.json",
        json.dumps(
            {
                f"{names[0]}.jpg": "A denim outfit.",
                f"{names[1]}.jpg": ["A blouse.", "Striped lower."],
            }
        ),
    )
    write(raw / "parsing" / f"{names[1]}.png", "mask")
    return raw


def test_build_outfit_table_joins_and_derives_columns(tmp_path):
    raw = staged_data(tmp_path)
    frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")

    assert list(frame["image_id"]) == sorted(frame["image_id"])
    assert frame["image_id"].is_unique
    men = frame.iloc[0]
    women = frame.iloc[1]
    assert (men["garment_id"], men["gender"], men["category_from_filename"]) == (
        "MEN-Denim-id_00000001",
        "men",
        "Denim",
    )
    assert not men["has_parsing"]
    assert women["is_full_body"]
    assert women["caption"] == "A blouse. Striped lower."
    assert not frame[["garment_id", "gender", "category_from_filename"]].isna().any().any()


def test_malformed_product_filename_is_dropped_and_counted(tmp_path, caplog):
    raw = staged_data(tmp_path)
    additions = {
        "shape_anno_all.txt": "invalid.jpg " + "0 " * 11 + "0\n",
        "fabric_ann.txt": "invalid.jpg 0 1 7\n",
        "pattern_ann.txt": "invalid.jpg 3 2 7\n",
    }
    for name, row in additions.items():
        path = raw / "labels" / name
        path.write_text(path.read_text(encoding="utf-8") + row, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="rmo.data.build_dataset"):
        frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")

    assert "invalid" not in set(frame["image_id"])
    assert "dropped 1 rows with invalid product filenames" in caplog.text


def test_build_dataset_writes_reproducible_parquet(tmp_path):
    raw = staged_data(tmp_path)
    output = tmp_path / "processed" / "outfits.parquet"
    first = build_dataset(raw, output)
    first_bytes = output.read_bytes()
    second = build_dataset(raw, output)

    pd.testing.assert_frame_equal(first, second)
    assert output.read_bytes() == first_bytes
    assert len(pd.read_parquet(output)) == 2


def test_failed_parquet_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    raw = staged_data(tmp_path)
    output = tmp_path / "processed" / "outfits.parquet"

    def fail_write(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        build_dataset(raw, output)

    assert not list(output.parent.glob("*.tmp"))


def test_missing_caption_fails_loudly(tmp_path):
    raw = staged_data(tmp_path)
    captions = json.loads((raw / "captions.json").read_text(encoding="utf-8"))
    captions.pop(next(iter(captions)))
    (raw / "captions.json").write_text(json.dumps(captions), encoding="utf-8")

    with pytest.raises(DatasetError, match="Missing captions for 1 images"):
        build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")