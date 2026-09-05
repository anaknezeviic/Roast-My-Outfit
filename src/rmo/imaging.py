"""Image input normalisation for every model entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from rmo import paths

__all__ = ["IN_MEMORY_ID", "ImageInput", "image_identity", "load_image"]

ImageInput = Union[str, Path, Image.Image, np.ndarray]

IN_MEMORY_ID = "in_memory"


def image_identity(image: ImageInput) -> tuple[str, str]:
    """Return the image id and source path, empty path for in-memory images."""
    if isinstance(image, (str, Path)):
        return Path(image).stem, str(image)
    filename = getattr(image, "filename", "")
    if filename:
        return Path(filename).stem, str(filename)
    return IN_MEMORY_ID, ""


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
