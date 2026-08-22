"""Cover palette extraction, area fractions and colour naming."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rmo.scoring.palette import (
    PaletteEntry,
    _REFERENCE_RGB,
    extract_palette,
    nearest_color_name,
)
from rmo.schemas import ColorName


@pytest.fixture()
def two_regions():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:7, :, 0] = 255
    image[7:, :, 2] = 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7] = 1
    mask[7:] = 2
    return image, mask


def test_area_fractions_sum_to_one(two_regions) -> None:
    image, mask = two_regions
    entries = extract_palette(image, mask)
    assert sum(entry.area_fraction for entry in entries) == pytest.approx(1.0, abs=1e-6)


def test_entries_are_sorted_by_area_fraction_descending() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:1] = 1
    mask[1:6] = 2
    mask[6:] = 3
    entries = extract_palette(image, mask)
    fractions = [entry.area_fraction for entry in entries]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions == pytest.approx([0.5, 0.4, 0.1])


def test_regions_are_named_and_measured(two_regions) -> None:
    image, mask = two_regions
    entries = extract_palette(image, mask)
    assert [entry.name for entry in entries] == [ColorName.red, ColorName.blue]
    assert [entry.area_fraction for entry in entries] == pytest.approx([0.7, 0.3])


def test_mask_none_is_tagged_wholeimage(two_regions) -> None:
    image, _ = two_regions
    entries = extract_palette(image, None)
    assert entries
    assert all(entry.source == "wholeimage" for entry in entries)


def test_mask_is_tagged_mask(two_regions) -> None:
    image, mask = two_regions
    assert all(entry.source == "mask" for entry in extract_palette(image, mask))


def test_background_only_mask_returns_nothing(two_regions) -> None:
    image, _ = two_regions
    assert extract_palette(image, np.zeros((10, 10), dtype=np.uint8)) == []


def test_more_colors_than_regions(two_regions) -> None:
    image, mask = two_regions
    entries = extract_palette(image, mask, n_colors=9)
    assert len(entries) == 2


def test_fewer_colors_than_regions_renormalises(two_regions) -> None:
    image, mask = two_regions
    entries = extract_palette(image, mask, n_colors=1)
    assert len(entries) == 1
    assert entries[0].area_fraction == pytest.approx(1.0, abs=1e-6)


def test_mask_shape_must_match_the_image(two_regions) -> None:
    image, _ = two_regions
    with pytest.raises(ValueError, match="does not match image shape"):
        extract_palette(image, np.ones((4, 4), dtype=np.uint8))


def test_named_extremes() -> None:
    assert nearest_color_name((0.0, 0.0, 0.0)) is ColorName.black
    assert nearest_color_name((100.0, 0.0, 0.0)) is ColorName.white


def test_non_finite_lab_is_unknown() -> None:
    assert nearest_color_name((float("nan"), 0.0, 0.0)) is ColorName.unknown


def test_reference_table_covers_every_name() -> None:
    assert set(_REFERENCE_RGB) == set(ColorName) - {ColorName.unknown}


def test_signatures_are_stable() -> None:
    def extract_palette_reference(image, mask, *, n_colors: int = 3) -> list[PaletteEntry]: ...

    def nearest_color_name_reference(lab: tuple[float, float, float]) -> ColorName: ...

    assert inspect.signature(extract_palette) == inspect.signature(extract_palette_reference)
    assert inspect.signature(nearest_color_name) == inspect.signature(nearest_color_name_reference)
