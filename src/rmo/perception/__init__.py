"""Models that turn outfit photographs into structured descriptions."""

from rmo.perception.cnn_adapter import CNNPerception
from rmo.perception.dummy import DummyPerception
from rmo.perception.vlm import REGISTRY_NAME, SmolVLMPerception
from rmo.pipeline import register

register(DummyPerception.name, DummyPerception)
register(CNNPerception.name, CNNPerception)
register(REGISTRY_NAME, SmolVLMPerception)
