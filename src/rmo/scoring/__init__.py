"""Models that turn outfit descriptions into style-compatibility scores."""

from rmo.pipeline import register
from rmo.scoring.dummy import DummyScorer

register(DummyScorer.name, DummyScorer)
