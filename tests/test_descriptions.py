"""Check descriptions built from labels, parsing masks and pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rmo.data.descriptions import (
    SOURCE_MODEL,
    DescriptionError,
    describe_image,
    describe_split,
    load_outfit_table,
)
from rmo.data.preflight import PreflightError
from rmo.perception.enrichment import SLOT_CATEGORIES
from rmo.schemas import (
    ColorName,
    Fabric,
    GarmentSlot,
    LowerLength,
    Neckline,
    Pattern,
    Provenance,
    SleeveLength,
)

IMAGE_ID = "WOMEN-Tees-id_00000001-01_7_additional"

UPPER_LABEL = 1
OUTER_LABEL = 2
LOWER_LABEL = 3
FOOTWEAR_LABEL = 11

DEFAULTS = {
    "upper_fabric": "na",
    "upper_pattern": "na",
    "outer_fabric": "na",
    "outer_pattern": "na",
    "lower_fabric": "na",
    "lower_pattern": "na",
    "sleeve_length": "na",
    "lower_length": "na",
    "neckline": "na",
    "has_shape": False,
    "caption": "",
}


def row(**overrides: object) -> pd.Series:
    return pd.Series({**DEFAULTS, **overrides})


def stage_photo(root: Path, pixels: np.ndarray, image_id: str = IMAGE_ID) -> None:
    directory = root / "raw" / "images"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(directory / f"{image_id}.jpg", quality=100)


def stage_mask(root: Path, mask: np.ndarray, image_id: str = IMAGE_ID) -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(directory / f"{image_id}_segm.png")


def two_band_pixels() -> np.ndarray:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[:6, :, 0] = 255
    pixels[6:, :, 2] = 255
    return pixels


def two_band_mask(top: int = UPPER_LABEL, bottom: int = LOWER_LABEL) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:6] = top
    mask[6:] = bottom
    return mask


@pytest.fixture()
def staged(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    stage_mask(tmp_path, two_band_mask())
    return tmp_path


def test_the_mask_decides_which_slots_appear(staged) -> None:
    description = describe_image(IMAGE_ID, row())
    assert [garment.slot for garment in description.garments] == [
        GarmentSlot.upper,
        GarmentSlot.lower,
    ]


def test_garments_follow_slot_declaration_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:3] = LOWER_LABEL
    mask[3:6] = FOOTWEAR_LABEL
    mask[6:] = OUTER_LABEL
    stage_mask(tmp_path, mask)
    slots = [garment.slot for garment in describe_image(IMAGE_ID, row()).garments]
    assert slots == [GarmentSlot.outer, GarmentSlot.lower, GarmentSlot.footwear]
    assert slots == sorted(slots, key=list(GarmentSlot).index)


def test_texture_columns_fill_fabric_and_pattern(staged) -> None:
    description = describe_image(
        IMAGE_ID,
        row(upper_fabric="denim", upper_pattern="striped", lower_fabric="cotton"),
    )
    upper, lower = description.garments
    assert (upper.fabric, upper.pattern) == (Fabric.denim, Pattern.striped)
    assert (lower.fabric, lower.pattern) == (Fabric.cotton, Pattern.na)


def test_shape_lands_on_the_garments_it_describes(staged) -> None:
    description = describe_image(
        IMAGE_ID,
        row(has_shape=True, sleeve_length="long", neckline="round", lower_length="long"),
    )
    upper, lower = description.garments
    assert upper.sleeve_length is SleeveLength.long
    assert upper.neckline is Neckline.round
    assert upper.length is None
    assert lower.length is LowerLength.long
    assert lower.sleeve_length is None
    assert lower.neckline is None


def test_an_unannotated_row_carries_no_shape_at_all(staged) -> None:
    description = describe_image(
        IMAGE_ID,
        row(has_shape=False, sleeve_length="long", neckline="round", lower_length="long"),
    )
    for garment in description.garments:
        assert garment.sleeve_length is None
        assert garment.neckline is None
        assert garment.length is None


def test_na_shape_codes_survive_as_values(staged) -> None:
    upper = describe_image(IMAGE_ID, row(has_shape=True)).garments[0]
    assert upper.sleeve_length is SleeveLength.na
    assert upper.neckline is Neckline.na


def test_outer_takes_the_sleeve_when_there_is_no_upper(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    stage_mask(tmp_path, two_band_mask(top=OUTER_LABEL))
    description = describe_image(IMAGE_ID, row(has_shape=True, sleeve_length="short"))
    outer = description.garments[0]
    assert outer.slot is GarmentSlot.outer
    assert outer.sleeve_length is SleeveLength.short


def test_colour_is_measured_from_each_masked_region(staged) -> None:
    upper, lower = describe_image(IMAGE_ID, row()).garments
    assert upper.color is ColorName.red
    assert lower.color is ColorName.blue
    assert upper.color_lab_source == "mask"
    assert upper.area_fraction == pytest.approx(0.6)
    assert lower.area_fraction == pytest.approx(0.4)


def test_without_a_mask_slots_come_from_texture_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    description = describe_image(IMAGE_ID, row(upper_fabric="cotton", lower_pattern="floral"))
    assert [garment.slot for garment in description.garments] == [
        GarmentSlot.upper,
        GarmentSlot.lower,
    ]
    measured = [g for g in description.garments if g.color_lab_source is not None]
    assert [g.color_lab_source for g in measured] == ["wholeimage"]
    assert measured[0].area_fraction is None


def test_a_row_with_no_populated_slot_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    with pytest.raises(DescriptionError, match="No garment slot"):
        describe_image(IMAGE_ID, row())


def test_the_record_is_ground_truth_with_a_relative_path(staged) -> None:
    description = describe_image(IMAGE_ID, row(caption="a red tee"))
    assert description.provenance is Provenance.gt
    assert description.source_model == SOURCE_MODEL
    assert description.caption == "a red tee"
    assert description.image_path == f"data/raw/images/{IMAGE_ID}.jpg"
    assert ":" not in description.image_path


def test_refs_are_numbered_per_slot(staged) -> None:
    assert describe_image(IMAGE_ID, row()).refs() == ["upper_0", "lower_0"]


def test_categories_come_from_the_shared_slot_vocabulary(staged) -> None:
    garments = describe_image(IMAGE_ID, row()).garments
    assert [garment.category for garment in garments] == ["top", "bottom"]
    for garment in garments:
        assert garment.category == SLOT_CATEGORIES[garment.slot]


def test_accessory_categories_also_come_from_the_vocabulary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage_photo(tmp_path, two_band_pixels())
    stage_mask(tmp_path, two_band_mask(top=OUTER_LABEL, bottom=FOOTWEAR_LABEL))
    garments = describe_image(IMAGE_ID, row()).garments
    assert [garment.category for garment in garments] == ["outerwear", "footwear"]


def test_load_outfit_table_rejects_repeated_image_ids(tmp_path) -> None:
    path = tmp_path / "outfits.parquet"
    pd.DataFrame({"image_id": ["a", "a"], "caption": ["", ""]}).to_parquet(path, index=False)
    with pytest.raises(DescriptionError, match="repeats 1 image ids"):
        load_outfit_table(path)


def test_load_outfit_table_reports_an_absent_file(tmp_path) -> None:
    with pytest.raises(DescriptionError, match="build_dataset"):
        load_outfit_table(tmp_path / "absent.parquet")


def test_load_outfit_table_indexes_by_image_id(tmp_path) -> None:
    path = tmp_path / "outfits.parquet"
    pd.DataFrame({"image_id": ["b", "a"], "caption": ["", ""]}).to_parquet(path, index=False)
    frame = load_outfit_table(path)
    assert list(frame.index) == ["b", "a"]
    assert frame.loc["a", "image_id"] == "a"


def _write_split(root: Path, name: str, image_ids: list[str]) -> None:
    directory = root / "processed" / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")


def test_describe_split_preflights_before_building(staged) -> None:
    _write_split(staged, "val", [IMAGE_ID, "absent-id"])
    table = pd.DataFrame([row(), row()], index=[IMAGE_ID, "absent-id"])
    with pytest.raises(PreflightError, match="absent-id"):
        list(describe_split("val", table=table))


def test_describe_split_rejects_ids_outside_the_table(staged) -> None:
    _write_split(staged, "val", [IMAGE_ID])
    table = pd.DataFrame([row()], index=["other-id"])
    with pytest.raises(DescriptionError, match="absent from the outfit table"):
        list(describe_split("val", table=table))


def test_describe_split_honours_sorted_order_and_limit(staged) -> None:
    second = "WOMEN-Tees-id_00000002-01_7_additional"
    stage_photo(staged, two_band_pixels(), image_id=second)
    stage_mask(staged, two_band_mask(), image_id=second)
    _write_split(staged, "val", [second, IMAGE_ID])
    table = pd.DataFrame([row(), row()], index=[second, IMAGE_ID])
    assert [d.image_id for d in describe_split("val", table=table)] == [IMAGE_ID, second]
    assert [d.image_id for d in describe_split("val", limit=1, table=table)] == [IMAGE_ID]
