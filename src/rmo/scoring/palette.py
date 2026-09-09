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
    "quantize_color_name",
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


# Garment-photo colour naming deliberately separates *how chromatic* a LAB
# centroid is from *which hue* it has.  A single nearest-reference CIE76
# search tends to collapse muted real-world colours onto gray/beige/black
# because the chromatic reference swatches are maximally saturated.
#
# These thresholds were chosen conservatively from the train-only colour
# audit: black assignments had p95 C* ~= 15, the beige reference has
# C* ~= 13, while visibly chromatic false-neutral examples started above
# roughly C*=20-25.  They are intentionally coarse vocabulary boundaries,
# not claims about universal colour perception.
_NAME_ACHROMATIC_CHROMA = 18.0
_NAME_BLACK_MAX_LIGHTNESS = 20.0
_NAME_WHITE_MIN_LIGHTNESS = 90.0
_NAME_BEIGE_MIN_LIGHTNESS = 70.0
_NAME_BEIGE_MAX_CHROMA = 28.0
_NAME_BROWN_MAX_LIGHTNESS = 50.0
_NAME_BROWN_MAX_CHROMA = 75.0
_NAME_NAVY_MAX_LIGHTNESS = 30.0
_NAME_NAVY_MAX_CHROMA = 95.0

_HUE_NAMES: tuple[ColorName, ...] = (
    ColorName.red,
    ColorName.orange,
    ColorName.yellow,
    ColorName.chartreuse,
    ColorName.green,
    ColorName.spring_green,
    ColorName.cyan,
    ColorName.azure,
    ColorName.blue,
    ColorName.violet,
    ColorName.magenta,
    ColorName.rose,
)


def nearest_color_name(lab: tuple[float, float, float]) -> ColorName:
    """Return the reference colour with the smallest CIE76 distance to ``lab``.

    This geometric helper is retained for diagnostics and compatibility.
    Garment palette entries use :func:`quantize_color_name`, whose neutral vs
    chromatic decision is more suitable for muted colours in photographs.
    """
    point = np.asarray(lab, dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return ColorName.unknown
    return _REFERENCE_NAMES[int(np.argmin(np.linalg.norm(_REFERENCE_LAB - point, axis=1)))]


def _lab_chroma(lab: tuple[float, float, float]) -> float:
    return float(np.hypot(float(lab[1]), float(lab[2])))


def _lab_hue_degrees(lab: tuple[float, float, float]) -> float:
    return float(np.degrees(np.arctan2(float(lab[2]), float(lab[1]))) % 360.0)


def _angular_distance(first: float, second: float) -> float:
    delta = abs(first - second) % 360.0
    return min(delta, 360.0 - delta)


def _reference_hue(name: ColorName) -> float:
    row = _REFERENCE_LAB[_REFERENCE_NAMES.index(name)]
    return _lab_hue_degrees((float(row[0]), float(row[1]), float(row[2])))


_REFERENCE_HUES: dict[ColorName, float] = {name: _reference_hue(name) for name in _HUE_NAMES}


def _hue_color_name(lab: tuple[float, float, float]) -> ColorName:
    hue = _lab_hue_degrees(lab)
    return min(
        _HUE_NAMES,
        key=lambda name: (_angular_distance(hue, _REFERENCE_HUES[name]), name.value),
    )


def _achromatic_name(lightness: float) -> ColorName:
    if lightness <= _NAME_BLACK_MAX_LIGHTNESS:
        return ColorName.black
    if lightness >= _NAME_WHITE_MIN_LIGHTNESS:
        return ColorName.white
    return ColorName.gray


def quantize_color_name(lab: tuple[float, float, float]) -> ColorName:
    """Map a measured LAB centroid onto RMO's coarse garment-colour vocabulary.

    The classifier first distinguishes low-chroma neutrals from chromatic
    colours.  Chromatic colours are named by LAB hue angle rather than by full
    CIE76 distance, so light or muted blue does not become gray merely because
    the reference blue swatch is darker and much more saturated.  Beige, brown
    and navy are handled as light/dark variants of their neighbouring hues.
    """
    point = np.asarray(lab, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return ColorName.unknown

    value = (float(point[0]), float(point[1]), float(point[2]))
    lightness = value[0]
    chroma = _lab_chroma(value)
    family = _hue_color_name(value)

    # Warm, light, weakly chromatic colours are represented by beige rather
    # than white/gray.  Requiring some chroma avoids turning neutral white into
    # beige because hue is unstable close to the neutral axis.
    if (
        5.0 <= chroma <= _NAME_BEIGE_MAX_CHROMA
        and lightness >= _NAME_BEIGE_MIN_LIGHTNESS
        and family in {ColorName.orange, ColorName.yellow}
    ):
        return ColorName.beige

    if chroma < _NAME_ACHROMATIC_CHROMA:
        return _achromatic_name(lightness)

    # Brown and navy are useful coarse names for sufficiently dark warm/cool
    # colours.  High-lightness members of the same hue families remain their
    # chromatic names (orange/yellow and azure/blue/violet respectively).
    if (
        lightness <= _NAME_BROWN_MAX_LIGHTNESS
        and chroma <= _NAME_BROWN_MAX_CHROMA
        and family in {ColorName.orange, ColorName.yellow}
    ):
        return ColorName.brown
    if (
        lightness <= _NAME_NAVY_MAX_LIGHTNESS
        and chroma <= _NAME_NAVY_MAX_CHROMA
        and family in {ColorName.azure, ColorName.blue, ColorName.violet}
    ):
        return ColorName.navy

    return family


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
            name=quantize_color_name(centroid),
            area_fraction=mass / retained,
            source=source,
        )
        for mass, _, centroid in kept
    ]

