"""Roast stage that replays the committed fixture roasts."""

from __future__ import annotations

from rmo import fixtures
from rmo.roast.base import RoastGenerator
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

__all__ = ["DummyRoaster"]


class DummyRoaster(RoastGenerator):
    """Replay the committed roast for one scored outfit."""

    name = "dummy_roaster"

    def __init__(self) -> None:
        self._records = fixtures.load_records("roast_outputs.jsonl", RoastOutput)

    def generate(self, description: OutfitDescription, score: OutfitScore) -> RoastOutput:
        record = self._records[description.image_id].model_copy(deep=True)
        record.provenance = description.provenance
        return record
