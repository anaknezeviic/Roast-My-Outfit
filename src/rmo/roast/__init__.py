"""Generators that turn a scored outfit into the text the product returns."""

from rmo.pipeline import register
from rmo.roast.dummy import DummyRoaster

register(DummyRoaster.name, DummyRoaster)
