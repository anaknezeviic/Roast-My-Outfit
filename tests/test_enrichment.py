"""Cover the mask and palette mechanics shared by every perception model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rmo.config import load_perception_config
from rmo.perception.enrichment import (
    SLOT_CATEGORIES,
    SLOT_PRIORITY,
    apply_palette,
    load_mask,
    mask_skeleton,
    slot_labels,
)
from rmo.schemas import ColorName, Garment, GarmentSlot, OutfitDescription


def write_mask(root: Path, image_id: str, array: np.ndarray) -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(directory / f"{image_id}_segm.png")


@pytest.fixture()
def two_regions() -> tuple[Image.Image, np.ndarray]:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[:7, :, 0] = 255
    pixels[7:, :, 2] = 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7] = 1
    mask[7:] = 5
    return Image.fromarray(pixels), mask


@pytest.fixture()
def labels() -> dict[GarmentSlot, frozenset[int]]:
    return slot_labels(load_perception_config())


def test_slot_priority_covers_every_slot() -> None:
    assert set(SLOT_PRIORITY) == set(GarmentSlot)
    assert len(SLOT_PRIORITY) == len(GarmentSlot)


def test_slot_categories_cover_every_slot() -> None:
    assert list(SLOT_CATEGORIES) == list(GarmentSlot)


@pytest.mark.parametrize("slot", list(GarmentSlot))
def test_slot_categories_are_valid_garment_categories(slot: GarmentSlot) -> None:
    category = SLOT_CATEGORIES[slot]
    assert 1 <= len(category) <= 64
    assert Garment(slot=slot, category=category).category == category


def test_slot_labels_cover_every_slot(labels) -> None:
    assert set(labels) == set(GarmentSlot)
    assert labels[GarmentSlot.other] == frozenset()


def test_load_mask_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    assert load_mask("missing", (10, 10), frozenset({1})) is None


def test_load_mask_rejects_a_shape_mismatch(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    write_mask(tmp_path, "small", np.ones((4, 4), dtype=np.uint8))
    with caplog.at_level(logging.WARNING, logger="rmo.perception.enrichment"):
        assert load_mask("small", (10, 10), frozenset({1})) is None
    assert len(caplog.records) == 1


def test_load_mask_zeroes_unclaimed_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.zeros((6, 6), dtype=np.uint8)
    array[:2] = 1
    array[2:4] = 13
    array[4:] = 15
    write_mask(tmp_path, "mixed", array)
    mask = load_mask("mixed", (6, 6), frozenset({1}))
    assert mask is not None
    assert set(np.unique(mask).tolist()) == {0, 1}


def test_load_mask_returns_none_when_nothing_is_claimed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.full((6, 6), 13, dtype=np.uint8)
    array[3:] = 15
    write_mask(tmp_path, "skin", array)
    assert load_mask("skin", (6, 6), frozenset({1})) is None


def test_load_mask_takes_the_first_channel_of_an_rgb_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.zeros((6, 6, 3), dtype=np.uint8)
    array[..., 0] = 2
    write_mask(tmp_path, "rgb", array)
    mask = load_mask("rgb", (6, 6), frozenset({2}))
    assert mask is not None
    assert mask.ndim == 2
    assert set(np.unique(mask).tolist()) == {2}


def test_mask_skeleton_follows_slot_declaration_order(labels) -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[:3] = 11
    mask[3:6] = 3
    mask[6:] = 1
    assert mask_skeleton(mask, labels) == [
        (GarmentSlot.upper, 1),
        (GarmentSlot.lower, 3),
        (GarmentSlot.footwear, 11),
    ]


def test_mask_skeleton_keeps_the_largest_class_of_a_shared_slot(labels) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:3] = 3
    mask[3:] = 6
    assert mask_skeleton(mask, labels) == [(GarmentSlot.lower, 6)]


def test_mask_skeleton_breaks_area_ties_towards_the_smallest_class(labels) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 6
    mask[5:] = 3
    assert mask_skeleton(mask, labels) == [(GarmentSlot.lower, 3)]


def test_mask_skeleton_skips_slots_with_no_pixels(labels) -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[:2] = 1
    skeleton = mask_skeleton(mask, labels)
    assert [slot for slot, _ in skeleton] == [GarmentSlot.upper]


def test_mask_skeleton_is_empty_for_an_unclaimed_mask(labels) -> None:
    assert mask_skeleton(np.zeros((4, 4), dtype=np.uint8), labels) == []


def test_apply_palette_joins_mask_regions_to_slots(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.lower, category="jeans"),
    ]
    apply_palette(garments, image, mask, labels)
    assert [garment.color for garment in garments] == [ColorName.red, ColorName.blue]
    assert all(garment.color_lab_source == "mask" for garment in garments)
    assert [garment.area_fraction for garment in garments] == pytest.approx([0.7, 0.3])


def test_apply_palette_area_fractions_sum_to_one(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.lower, category="jeans"),
    ]
    apply_palette(garments, image, mask, labels)
    total = sum(garment.area_fraction for garment in garments)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_apply_palette_skips_slots_with_no_mask_pixels(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.footwear, category="boots"),
    ]
    apply_palette(garments, image, mask, labels)
    boots = garments[1]
    assert boots.color_lab is None
    assert boots.color_lab_source is None
    assert boots.area_fraction is None


def test_apply_palette_shares_one_region_between_two_uppers(two_regions, labels) -> None:
    image, mask = two_regions
    description = OutfitDescription(
        image_id="fixture_000",
        source_model="fixture",
        garments=[
            Garment(slot=GarmentSlot.upper, category="tee"),
            Garment(slot=GarmentSlot.upper, category="cardigan"),
        ],
    )
    apply_palette(description.garments, image, mask, labels)
    first, second = description.garments
    assert first.color_lab == second.color_lab
    assert description.refs() == ["upper_0", "upper_1"]


def test_apply_palette_keeps_a_parsed_colour(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.lower, category="jeans", color=ColorName.red)]
    apply_palette(garments, image, mask, labels)
    assert garments[0].color is ColorName.red
    assert garments[0].color_lab is not None


def test_apply_palette_fills_only_unknown_colours(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.lower, category="jeans")]
    apply_palette(garments, image, mask, labels)
    assert garments[0].color is ColorName.blue


def test_apply_palette_without_a_mask_colours_the_priority_garment(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [
        Garment(slot=GarmentSlot.footwear, category="boots"),
        Garment(slot=GarmentSlot.dress, category="gown"),
    ]
    apply_palette(garments, image, None, labels)
    assert garments[1].color_lab_source == "wholeimage"
    assert garments[0].color_lab_source is None
    assert garments[0].color_lab is None


def test_apply_palette_without_a_mask_leaves_area_fraction_none(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [Garment(slot=GarmentSlot.dress, category="gown")]
    apply_palette(garments, image, None, labels)
    assert garments[0].area_fraction is None


def test_apply_palette_breaks_priority_ties_by_order(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.upper, category="cardigan"),
    ]
    apply_palette(garments, image, None, labels)
    assert garments[0].color_lab is not None
    assert garments[1].color_lab is None


def test_apply_palette_on_a_fallback_description_does_not_raise(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)]
    apply_palette(garments, image, mask, labels)
    assert garments[0].color_lab is None
    assert garments[0].color is ColorName.unknown
