"""Dominant garment colours in CIELAB and the coarse names they quantise onto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rmo.imaging import load_image
from rmo.schemas import ColorLabSource, ColorName

__all__ = ["PaletteEntry", "extract_palette", "nearest_color_name"]

_D65_WHITE = np.array([0.95047, 1.0, 1.08883])

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

_REFERENCE_RGB: dict[ColorName, tuple[int, int, int]] = {
    ColorName.red: (255, 0, 0),
    ColorName.orange: (255, 128, 0),
    ColorName.yellow: (255, 255, 0),
    ColorName.chartreuse: (128, 255, 0),
    ColorName.green: (0, 255, 0),
    ColorName.spring_green: (0, 255, 128),
    ColorName.cyan: (0, 255, 255),
    ColorName.azure: (0, 128, 255),
    ColorName.blue: (0, 0, 255),
    ColorName.violet: (128, 0, 255),
    ColorName.magenta: (255, 0, 255),
    ColorName.rose: (255, 0, 128),
    ColorName.black: (0, 0, 0),
    ColorName.white: (255, 255, 255),
    ColorName.gray: (128, 128, 128),
    ColorName.beige: (245, 245, 220),
    ColorName.brown: (139, 69, 19),
    ColorName.navy: (0, 0, 128),
}


@dataclass(frozen=True, slots=True)
class PaletteEntry:
    """One dominant colour of an image region."""

    lab: tuple[float, float, float]
    name: ColorName
    area_fraction: float
    source: ColorLabSource


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` array of 0-255 sRGB values to CIELAB under D65."""
    channels = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    xyz = (linear @ _RGB_TO_XYZ.T) / _D65_WHITE
    scaled = np.where(xyz > 216 / 24389, np.cbrt(xyz), (xyz * 24389 / 27 + 16) / 116)
    return np.stack(
        (
            116 * scaled[..., 1] - 16,
            500 * (scaled[..., 0] - scaled[..., 1]),
            200 * (scaled[..., 1] - scaled[..., 2]),
        ),
        axis=-1,
    )


_REFERENCE_NAMES: tuple[ColorName, ...] = tuple(_REFERENCE_RGB)

_REFERENCE_LAB = _srgb_to_lab(np.array([_REFERENCE_RGB[name] for name in _REFERENCE_NAMES]))


def nearest_color_name(lab: tuple[float, float, float]) -> ColorName:
    """Return the reference colour with the smallest CIE76 distance to ``lab``."""
    point = np.asarray(lab, dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return ColorName.unknown
    return _REFERENCE_NAMES[int(np.argmin(np.linalg.norm(_REFERENCE_LAB - point, axis=1)))]


def extract_palette(image, mask, *, n_colors: int = 3) -> list[PaletteEntry]:
    """Return the dominant colours of ``image``, ordered by descending area fraction."""
    pixels = np.asarray(load_image(image), dtype=np.uint8)
    shape = pixels.shape[:2]

    if mask is None:
        labels = np.ones(shape, dtype=np.uint8)
        source: ColorLabSource = "wholeimage"
    else:
        labels = np.asarray(mask)
        if labels.shape != shape:
            raise ValueError(f"Mask shape {labels.shape} does not match image shape {shape}.")
        source = "mask"

    foreground = labels != 0
    if not foreground.any():
        return []

    lab = _srgb_to_lab(pixels)
    regions = [(int(np.count_nonzero(labels == label)), label) for label in np.unique(labels[foreground])]
    regions.sort(key=lambda region: -region[0])

    kept = regions[:n_colors]
    total = sum(count for count, _ in kept)

    entries = []
    for count, label in kept:
        mean = lab[labels == label].mean(axis=0)
        centroid = (float(mean[0]), float(mean[1]), float(mean[2]))
        entries.append(
            PaletteEntry(
                lab=centroid,
                name=nearest_color_name(centroid),
                area_fraction=count / total,
                source=source,
            )
        )
    return entries
