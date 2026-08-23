"""Perception stage that replays the committed fixture descriptions."""

from __future__ import annotations

from pathlib import Path

from rmo import fixtures
from rmo.imaging import ImageInput
from rmo.perception.base import PerceptionModel
from rmo.schemas import OutfitDescription

__all__ = ["DummyPerception"]


def _image_id_for(image: ImageInput) -> str:
    """Return the fixture id named by an image path or by an image opened from one."""
    if isinstance(image, (str, Path)):
        return Path(image).stem
    filename = getattr(image, "filename", "")
    if filename:
        return Path(filename).stem
    raise TypeError(f"Fixture replay needs an image path; got {type(image).__name__}.")


class DummyPerception(PerceptionModel):
    """Replay the committed description for one image."""

    name = "dummy_perception"

    def __init__(self) -> None:
        self._records = fixtures.load_records("outfit_descriptions.jsonl", OutfitDescription)

    def predict(self, image: ImageInput) -> OutfitDescription:
        return self._records[_image_id_for(image)].model_copy(deep=True)
