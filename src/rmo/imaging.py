"""Image input normalisation for every model entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from rmo import paths

__all__ = ["ImageInput", "load_image"]

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def load_image(image: ImageInput) -> Image.Image:
    """Return ``image`` as an RGB image, resolving relative paths against the repo root."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Array input must be HWC uint8; got {image.ndim}D {image.dtype}."
            )
        return Image.fromarray(image).convert("RGB")

    path = Path(image)
    if not path.is_absolute():
        path = paths.repo_root() / path
    with Image.open(path) as handle:
        return handle.convert("RGB")
