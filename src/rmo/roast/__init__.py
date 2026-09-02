"""Generators that turn a scored outfit into the text the product returns."""

from rmo.pipeline import register
from rmo.roast.dummy import DummyRoaster
from rmo.roast.rules import RuleBasedRoaster

register(DummyRoaster.name, DummyRoaster)
register(RuleBasedRoaster.name, RuleBasedRoaster)