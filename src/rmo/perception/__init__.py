"""Models that turn outfit photographs into structured descriptions."""

from rmo.perception.dummy import DummyPerception
from rmo.pipeline import register

register(DummyPerception.name, DummyPerception)
