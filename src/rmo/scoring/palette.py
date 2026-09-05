"""Dominant garment colours in CIELAB and the coarse names they quantise onto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rmo.config import ConfigError, load_scoring_config
from rmo.imaging import load_image
from rmo.schemas import ColorLabSource, ColorName

__all__ = [
    "PaletteEntry",
    "extract_palette",
    "mean_lab",
    "nearest_color_name",
    "reference_lab",
]

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


def reference_lab(name: ColorName) -> tuple[float, float, float] | None:
    """Return the reference CIELAB value of a named colour, ``None`` for ``unknown``."""
    if name not in _REFERENCE_RGB:
        return None
    row = _REFERENCE_LAB[_REFERENCE_NAMES.index(name)]
    return (float(row[0]), float(row[1]), float(row[2]))


def mean_lab(pixels: np.ndarray) -> tuple[float, float, float]:
    """Return the mean CIELAB value of an ``(..., 3)`` array of 0-255 sRGB pixels."""
    array = np.asarray(pixels)
    if array.shape[-1:] != (3,):
        raise ValueError(
            f"Pixels must have three channels on the last axis; got shape {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Pixels must be integers in [0, 255]; got dtype {array.dtype}.")
    if array.size and (array.min() < 0 or array.max() > 255):
        raise ValueError("Pixels must be integers in [0, 255]; got a value outside that range.")
    values = _srgb_to_lab(array.astype(np.uint8)).reshape(-1, 3)
    if not len(values):
        raise ValueError("Cannot average an empty pixel array.")
    mean = values.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


@dataclass(frozen=True, slots=True)
class _PaletteSettings:
    """Clustering budget and filtering thresholds read from the scoring config."""

    seed: int
    n_clusters: int
    n_init: int
    min_area_fraction: float
    max_fit_pixels: int


def _positive_int(section: dict, key: str) -> int:
    """Return a strictly positive integer option."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"palette.{key} must be a positive integer; got {value!r}.")
    return value


def _palette_settings() -> _PaletteSettings:
    """Return the validated ``palette`` section of the scoring configuration."""
    section = load_scoring_config().get("palette")
    if not isinstance(section, dict):
        raise ConfigError("Scoring configuration has no palette section.")

    seed = section.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"palette.seed must be a non-negative integer; got {seed!r}.")

    floor = section.get("min_area_fraction")
    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not 0.0 <= floor < 1.0
    ):
        raise ConfigError(f"palette.min_area_fraction must be in [0, 1); got {floor!r}.")

    return _PaletteSettings(
        seed=seed,
        n_clusters=_positive_int(section, "n_clusters"),
        n_init=_positive_int(section, "n_init"),
        min_area_fraction=float(floor),
        max_fit_pixels=_positive_int(section, "max_fit_pixels"),
    )


def _fit_sample(values: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Return at most ``budget`` rows of ``values``, chosen deterministically."""
    if len(values) <= budget:
        return values
    # k-means centres depend on row order, so the drawn index is sorted before slicing.
    index = np.sort(np.random.default_rng(seed).choice(len(values), size=budget, replace=False))
    return values[index]


def _cluster_region(region: np.ndarray, settings: _PaletteSettings) -> list[tuple[int, np.ndarray]]:
    """Return the population and centroid of every nonempty cluster of one region."""
    from sklearn.cluster import KMeans

    sample = _fit_sample(region, settings.max_fit_pixels, settings.seed)
    k = min(settings.n_clusters, len(np.unique(sample, axis=0)))
    model = KMeans(n_clusters=k, n_init=settings.n_init, random_state=settings.seed).fit(sample)
    assigned = model.predict(region)
    clusters = []
    for cid in range(k):
        members = region[assigned == cid]
        if len(members):
            clusters.append((len(members), members.mean(axis=0)))
    return clusters


def extract_palette(image, mask, *, n_colors: int = 3) -> list[PaletteEntry]:
    """Return the dominant colours of ``image``, ordered by descending area fraction."""
    if isinstance(n_colors, bool) or not isinstance(n_colors, int) or n_colors < 1:
        raise ValueError(f"n_colors must be a positive integer; got {n_colors!r}.")

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

    settings = _palette_settings()
    total = int(np.count_nonzero(foreground))

    candidates = []
    for label in np.unique(labels[foreground]):
        region = _srgb_to_lab(pixels[labels == label])
        for mass, centroid in _cluster_region(region, settings):
            candidates.append(
                (mass, int(label), (float(centroid[0]), float(centroid[1]), float(centroid[2])))
            )

    kept = [item for item in candidates if item[0] / total >= settings.min_area_fraction]
    if not kept:
        kept = [max(candidates, key=lambda item: item[0])]

    kept.sort(key=lambda item: (-item[0], item[1], item[2]))
    kept = kept[:n_colors]
    retained = sum(mass for mass, _, _ in kept)

    return [
        PaletteEntry(
            lab=centroid,
            name=nearest_color_name(centroid),
            area_fraction=mass / retained,
            source=source,
        )
        for mass, _, centroid in kept
    ]

