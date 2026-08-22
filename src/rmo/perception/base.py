"""Interface every perception model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rmo.imaging import ImageInput
from rmo.schemas import OutfitDescription

__all__ = ["PerceptionModel"]


class PerceptionModel(ABC):
    """Turn an image into an outfit description."""

    name: str

    @abstractmethod
    def predict(self, image: ImageInput) -> OutfitDescription:
        """Return the description of the outfit in one image."""

    def predict_batch(self, images: Sequence[ImageInput]) -> list[OutfitDescription]:
        """Return one description per image, in input order."""
        return [self.predict(image) for image in images]
