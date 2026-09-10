"""Models that turn outfit photographs into structured descriptions."""

from rmo.perception.cnn_adapter import CNNPerception
from rmo.perception.dummy import DummyPerception
from rmo.perception.vlm import (
    DEFAULT_ADAPTER_PATH,
    FINETUNED_REGISTRY_NAME,
    REGISTRY_NAME,
    SmolVLMPerception,
)
from rmo.pipeline import register


def _fine_tuned_smolvlm() -> SmolVLMPerception:
    return SmolVLMPerception(
        adapter_path=DEFAULT_ADAPTER_PATH,
    )


register(DummyPerception.name, DummyPerception)
register(CNNPerception.name, CNNPerception)
register(REGISTRY_NAME, SmolVLMPerception)
register(FINETUNED_REGISTRY_NAME, _fine_tuned_smolvlm)