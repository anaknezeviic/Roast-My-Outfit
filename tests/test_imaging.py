"""Cover image input normalisation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rmo.imaging import load_image


@pytest.fixture()
def png(tmp_path):
    path = tmp_path / "swatch.png"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(path)
    return path


def test_path_input(png) -> None:
    loaded = load_image(png)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (4, 3)


def test_str_input(png) -> None:
    assert load_image(str(png)).mode == "RGB"


def test_pil_input_is_converted_to_rgb() -> None:
    loaded = load_image(Image.new("L", (2, 2), 128))
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (128, 128, 128)


def test_array_input_keeps_its_pixels() -> None:
    array = np.zeros((3, 4, 3), dtype=np.uint8)
    array[..., 0] = 255
    loaded = load_image(array)
    assert loaded.mode == "RGB"
    assert loaded.size == (4, 3)
    assert loaded.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 3), dtype=np.float32),
    ],
)
def test_array_input_rejects_anything_but_hwc_uint8(array) -> None:
    with pytest.raises(ValueError):
        load_image(array)
