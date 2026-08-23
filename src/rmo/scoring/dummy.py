"""Scoring stage that replays the committed fixture scores."""

from __future__ import annotations

from rmo import fixtures
from rmo.schemas import OutfitDescription, OutfitScore
from rmo.scoring.base import ScoringModel

__all__ = ["DummyScorer"]


class DummyScorer(ScoringModel):
    """Replay the committed score for one outfit description."""

    name = "dummy_scorer"

    def __init__(self) -> None:
        self._records = fixtures.load_records("outfit_scores.jsonl", OutfitScore)

    def score(self, description: OutfitDescription) -> OutfitScore:
        record = self._records[description.image_id].model_copy(deep=True)
        record.provenance = description.provenance
        return record
