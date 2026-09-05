"""Models that turn outfit descriptions into style-compatibility scores."""

from rmo.pipeline import register
from rmo.scoring.dummy import DummyScorer
from rmo.scoring.rules import RuleScorer

register(DummyScorer.name, DummyScorer)
register(RuleScorer.name, RuleScorer)
