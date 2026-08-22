"""Interface every roast generator implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

__all__ = ["RoastGenerator"]


class RoastGenerator(ABC):
    """Turn an outfit description and its score into a roast."""

    name: str

    @abstractmethod
    def generate(self, description: OutfitDescription, score: OutfitScore) -> RoastOutput:
        """Return the roast for one scored outfit."""
