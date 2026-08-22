"""Interface every scoring model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rmo.schemas import OutfitDescription, OutfitScore

__all__ = ["ScoringModel"]


class ScoringModel(ABC):
    """Turn an outfit description into a score."""

    name: str

    @abstractmethod
    def score(self, description: OutfitDescription) -> OutfitScore:
        """Return the score for one outfit description."""
