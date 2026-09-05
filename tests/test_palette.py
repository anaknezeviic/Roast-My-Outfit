"""Cover palette extraction, area fractions and colour naming."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from rmo.config import ConfigError
from rmo.scoring.palette import (
    PaletteEntry,
    _REFERENCE_RGB,
    _fit_sample,
    _srgb_to_lab,
    extract_palette,
    mean_lab,
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


@pytest.fixture()
def three_blocks():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5, :, 0] = 255
    image[5:8, :, 1] = 255
    image[8:, :, 2] = 255
    return image


@pytest.fixture()
def dithered():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:7, :, 0] = 200
    image[7:, :, 2] = 200
    noise = np.random.default_rng(7).integers(-4, 5, size=image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_three_colour_blocks_return_their_three_names(three_blocks) -> None:
    entries = extract_palette(three_blocks, None, n_colors=3)
    assert [entry.name for entry in entries] == [ColorName.red, ColorName.green, ColorName.blue]
    assert [entry.area_fraction for entry in entries] == pytest.approx([0.5, 0.3, 0.2], abs=0.02)


def test_repeated_calls_return_identical_labs(three_blocks) -> None:
    first = [entry.lab for entry in extract_palette(three_blocks, None)]
    second = [entry.lab for entry in extract_palette(three_blocks, None)]
    assert first == second


def test_an_all_grey_image_is_neutral_and_finite() -> None:
    entries = extract_palette(np.full((10, 10, 3), 128, dtype=np.uint8), None)
    assert [entry.name for entry in entries] == [ColorName.gray]
    assert all(np.isfinite(entries[0].lab))
    assert entries[0].area_fraction == pytest.approx(1.0)


def test_a_single_colour_label_yields_one_entry() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    assert len(extract_palette(image, np.ones((10, 10), dtype=np.uint8), n_colors=3)) == 1


def test_equal_colours_in_three_labels_stay_separate() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 1
    mask[5:8] = 2
    mask[8:] = 3
    entries = extract_palette(image, mask)
    assert [entry.name for entry in entries] == [ColorName.red] * 3
    assert [entry.area_fraction for entry in entries] == pytest.approx([0.5, 0.3, 0.2])


def test_a_binary_mask_at_one_colour_returns_the_largest_cluster(two_regions) -> None:
    image, _ = two_regions
    binary = np.ones((10, 10), dtype=np.uint8)
    entries = extract_palette(image, binary, n_colors=1)
    assert len(entries) == 1
    assert entries[0].name is ColorName.red
    assert entries[0].lab != mean_lab(image)


def test_a_binary_mask_with_three_colours_returns_all_three(three_blocks) -> None:
    binary = np.ones((10, 10), dtype=np.uint8)
    entries = extract_palette(three_blocks, binary, n_colors=3)
    assert [entry.name for entry in entries] == [ColorName.red, ColorName.green, ColorName.blue]


def test_clusters_below_the_area_floor_are_dropped() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    flat = image.reshape(-1, 3)
    flat[:96, 0] = 255
    flat[96:99, 1] = 255
    flat[99:, 2] = 255
    entries = extract_palette(image, None)
    assert [entry.name for entry in entries] == [ColorName.red, ColorName.green]
    assert sum(entry.area_fraction for entry in entries) == pytest.approx(1.0)


def test_filtering_never_empties_a_nonempty_foreground(scoring_config, three_blocks) -> None:
    scoring_config(palette={"min_area_fraction": 0.99})
    entries = extract_palette(three_blocks, None)
    assert len(entries) == 1
    assert entries[0].area_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("n_colors", [0, -1, 2.5, True])
def test_a_nonpositive_or_fractional_colour_count_raises(two_regions, n_colors) -> None:
    image, mask = two_regions
    with pytest.raises(ValueError, match="n_colors must be a positive integer"):
        extract_palette(image, mask, n_colors=n_colors)


def test_an_out_of_range_area_floor_raises_a_config_error(scoring_config, three_blocks) -> None:
    scoring_config(palette={"min_area_fraction": 1.0})
    with pytest.raises(ConfigError, match=r"palette.min_area_fraction must be in \[0, 1\)"):
        extract_palette(three_blocks, None)


def test_a_nonpositive_cluster_budget_raises_a_config_error(scoring_config, three_blocks) -> None:
    scoring_config(palette={"n_clusters": 0})
    with pytest.raises(ConfigError, match="palette.n_clusters must be a positive integer"):
        extract_palette(three_blocks, None)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"seed": -1}, "palette.seed must be a non-negative integer"),
        ({"seed": True}, "palette.seed must be a non-negative integer"),
        ({"seed": 1.5}, "palette.seed must be a non-negative integer"),
        ({"n_clusters": True}, "palette.n_clusters must be a positive integer"),
        ({"n_init": 0}, "palette.n_init must be a positive integer"),
        ({"n_init": True}, "palette.n_init must be a positive integer"),
        ({"max_fit_pixels": 0}, "palette.max_fit_pixels must be a positive integer"),
        ({"max_fit_pixels": True}, "palette.max_fit_pixels must be a positive integer"),
    ],
)
def test_an_invalid_palette_option_raises_a_config_error(
    scoring_config, three_blocks, overrides, match
) -> None:
    scoring_config(palette=overrides)
    with pytest.raises(ConfigError, match=match):
        extract_palette(three_blocks, None)


def test_a_missing_palette_section_raises_a_config_error(scoring_config, three_blocks) -> None:
    scoring_config(palette=None)
    with pytest.raises(ConfigError, match="no palette section"):
        extract_palette(three_blocks, None)


def test_bounded_fitting_still_counts_every_foreground_pixel(scoring_config, two_regions) -> None:
    scoring_config(palette={"max_fit_pixels": 50})
    image, _ = two_regions
    entries = extract_palette(image, np.ones((10, 10), dtype=np.uint8), n_colors=2)
    assert [entry.name for entry in entries] == [ColorName.red, ColorName.blue]
    assert [entry.area_fraction for entry in entries] == pytest.approx([0.7, 0.3])


def test_the_fit_sample_follows_the_configured_seed() -> None:
    values = np.arange(300, dtype=np.float64).reshape(100, 3)
    drawn = _fit_sample(values, 20, 20260101)
    assert len(drawn) == 20
    assert np.array_equal(drawn, _fit_sample(values, 20, 20260101))
    assert not np.array_equal(drawn, _fit_sample(values, 20, 11))
    assert np.array_equal(drawn[:, 0], np.sort(drawn[:, 0]))
    assert np.array_equal(_fit_sample(values, 500, 20260101), values)


def test_the_cluster_budget_comes_from_the_config(scoring_config, three_blocks) -> None:
    scoring_config(palette={"n_clusters": 2})
    assert len(extract_palette(three_blocks, None)) == 2
    scoring_config(palette={"n_clusters": 3})
    shipped = extract_palette(three_blocks, None)
    assert len(shipped) == 3
    scoring_config(palette={"n_init": 1})
    assert extract_palette(three_blocks, None) == shipped


def test_a_region_with_fewer_pixels_than_clusters_yields_finite_labs() -> None:
    image = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    entries = extract_palette(image, np.ones((1, 2), dtype=np.uint8))
    assert len(entries) == 2
    assert all(np.isfinite(entry.lab).all() for entry in entries)


def test_clustering_emits_no_warnings() -> None:
    image = np.full((10, 10, 3), 90, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(extract_palette(image, np.ones((10, 10), dtype=np.uint8))) == 1


def test_a_multi_label_selection_ranks_clusters_by_pooled_mass() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:6, :, 0] = 200
    image[6:, :, 2] = 200
    noise = np.random.default_rng(3).integers(-40, 41, size=(6, 10, 3))
    image[:6] = np.clip(image[:6].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:6] = 1
    mask[6:] = 2
    entries = extract_palette(image, mask, n_colors=1)
    assert len(entries) == 1
    assert entries[0].name is ColorName.blue


def test_mean_lab_equals_a_lab_space_pixel_average(dithered) -> None:
    expected = _srgb_to_lab(dithered).reshape(-1, 3).mean(axis=0)
    assert mean_lab(dithered) == (float(expected[0]), float(expected[1]), float(expected[2]))


def test_mean_lab_differs_from_the_dominant_cluster(dithered) -> None:
    dominant = extract_palette(dithered, None, n_colors=1)[0]
    assert dominant.lab != mean_lab(dithered)


@pytest.mark.parametrize("shape", [(4, 2), (0, 3), (5,)])
def test_mean_lab_rejects_a_wrong_shaped_array(shape) -> None:
    with pytest.raises(ValueError):
        mean_lab(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize(
    "pixels",
    [
        np.array([[300, -5, 2]], dtype=np.int16),
        np.array([[-1, 0, 0]], dtype=np.int16),
        np.array([[256, 0, 0]], dtype=np.int32),
    ],
)
def test_mean_lab_rejects_values_outside_the_byte_range(pixels) -> None:
    with pytest.raises(ValueError, match=r"integers in \[0, 255\]"):
        mean_lab(pixels)


@pytest.mark.parametrize(
    "pixels",
    [
        np.array([[300.0, -5.0, 1.7]]),
        np.array([[10.5, 20.5, 30.5]], dtype=np.float32),
        np.array([[True, False, True]]),
    ],
)
def test_mean_lab_rejects_a_non_integer_dtype(pixels) -> None:
    with pytest.raises(ValueError, match=r"integers in \[0, 255\]"):
        mean_lab(pixels)

