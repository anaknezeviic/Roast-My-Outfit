"""Outfit table tests using local staged fixtures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from rmo.data.build_dataset import DatasetError, build_dataset, build_outfit_table
from rmo.data.parse_annotations import SHAPE_COLUMNS


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
    write(raw / "parsing" / f"{names[1]}_segm.png", "mask")
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


def test_texture_only_image_has_explicit_shape_values_and_blank_caption(tmp_path):
    raw = staged_data(tmp_path)
    image_id = "WOMEN-Dresses-id_00000003-01_1_front"
    for name, codes in (("fabric_ann.txt", "1 7 7"), ("pattern_ann.txt", "3 7 7")):
        path = raw / "labels" / name
        path.write_text(
            path.read_text(encoding="utf-8") + f"{image_id}.jpg {codes}\n",
            encoding="utf-8",
        )

    frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")
    row = frame.loc[frame["image_id"] == image_id].iloc[0]

    assert all(row[column] == "na" for column in (
        "sleeve_length",
        "lower_length",
        "socks",
        "hat",
        "glasses",
        "neckwear",
        "wrist",
        "ring",
        "waist_accessory",
        "neckline",
        "cardigan",
        "navel_covered",
    ))
    assert row["caption"] == ""


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


def test_a_mask_named_after_the_image_alone_also_counts(tmp_path):
    raw = staged_data(tmp_path)
    mask = next((raw / "parsing").glob("*.png"))
    mask.rename(mask.with_name(mask.name.replace("_segm", "")))

    frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")

    assert int(frame["has_parsing"].sum()) == 1


def test_missing_caption_becomes_an_empty_string(tmp_path, caplog):
    raw = staged_data(tmp_path)
    captions = json.loads((raw / "captions.json").read_text(encoding="utf-8"))
    dropped = next(iter(captions))
    captions.pop(dropped)
    (raw / "captions.json").write_text(json.dumps(captions), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="rmo.data.build_dataset"):
        frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")

    assert len(frame) == 2
    assert frame.loc[frame["image_id"] == Path(dropped).stem, "caption"].item() == ""
    assert "1 of 2 images carry no caption" in caplog.text


def test_a_caption_file_matching_nothing_fails_loudly(tmp_path):
    raw = staged_data(tmp_path)
    write(raw / "captions.json", json.dumps({"unrelated.jpg": "A hat."}))

    with pytest.raises(DatasetError, match="names a labeled image"):
        build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")


def test_images_without_a_shape_annotation_are_kept_and_flagged(tmp_path):
    raw = staged_data(tmp_path)
    shape = raw / "labels" / "shape_anno_all.txt"
    kept = shape.read_text(encoding="utf-8").splitlines()[0]
    write(shape, f"{kept}\n")

    frame = build_outfit_table(raw / "labels", raw / "captions.json", raw / "parsing")

    assert len(frame) == 2
    annotated = frame.loc[frame["has_shape"]]
    unannotated = frame.loc[~frame["has_shape"]]
    assert len(annotated) == 1 and len(unannotated) == 1
    assert set(unannotated[list(SHAPE_COLUMNS)].iloc[0]) == {"na"}
    assert unannotated["upper_fabric"].item() == "denim"