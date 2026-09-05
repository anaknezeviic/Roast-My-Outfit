"""Cover the SmolVLM entry point without importing torch."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from rmo.config import ConfigError, load_perception_config
from rmo.perception import vlm
from rmo.perception.base import PerceptionModel
from rmo.perception.vlm import DEFAULT_MODEL_ID, SmolVLMPerception
from rmo.schemas import ColorName

TORCH_PROBE = "import sys; import rmo.perception.vlm; print('torch' in sys.modules)"


def test_module_imports_without_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", TORCH_PROBE], check=True, capture_output=True, text=True
    )
    assert importlib.import_module("rmo.perception.vlm") is vlm
    assert completed.stdout.strip() == "False"


def test_constructor_does_not_need_torch() -> None:
    assert SmolVLMPerception().name == DEFAULT_MODEL_ID


def test_name_follows_the_checkpoint_id() -> None:
    assert SmolVLMPerception("org/other").name == "org/other"


def test_default_model_id_is_the_500m_checkpoint() -> None:
    assert DEFAULT_MODEL_ID == "HuggingFaceTB/SmolVLM-500M-Instruct"


def test_missing_config_raises_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        SmolVLMPerception(config_path=tmp_path / "absent.yaml")


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None, reason="transformers is installed"
)
def test_predict_reports_the_missing_dependencies() -> None:
    with pytest.raises(ImportError, match="torch and transformers"):
        SmolVLMPerception().predict(np.zeros((4, 4, 3), dtype=np.uint8))


def test_is_a_perception_model() -> None:
    assert issubclass(SmolVLMPerception, PerceptionModel)
    assert SmolVLMPerception.predict_batch is PerceptionModel.predict_batch


class StubProcessor:
    def __init__(self, torch) -> None:
        self._torch = torch
        self.messages: list[dict] = []
        self.images: list = []
        self.decoded: list = []

    def apply_chat_template(self, messages, add_generation_prompt):
        self.messages = messages
        assert add_generation_prompt is True
        return "PROMPT"

    def __call__(self, text, images, return_tensors):
        self.text = text
        self.images = images
        self.return_tensors = return_tensors
        return StubInputs(input_ids=self._torch.tensor([[11, 12, 13]]))

    def batch_decode(self, tokens, skip_special_tokens):
        self.decoded = tokens.tolist()
        assert skip_special_tokens is True
        return ["upper: tee, white, graphic, cotton, na"]


class StubInputs(dict):
    def to(self, device):
        self.device = device
        return self


class StubModel:
    def __init__(self, torch) -> None:
        self._torch = torch
        self.kwargs: dict = {}

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return self._torch.tensor([[11, 12, 13, 44, 55]])


@pytest.fixture()
def wired():
    torch = pytest.importorskip("torch")
    model = SmolVLMPerception()
    model._processor = StubProcessor(torch)
    model._model = StubModel(torch)
    model._device = "cpu"
    return model


def test_generate_returns_only_the_new_tokens(wired) -> None:
    text = wired._generate(Image.new("RGB", (4, 4)))
    assert wired._processor.decoded == [[44, 55]]
    assert text == "upper: tee, white, graphic, cotton, na"


def test_generate_sends_the_configured_prompt_with_the_image(wired) -> None:
    picture = Image.new("RGB", (4, 4))
    wired._generate(picture)
    content = wired._processor.messages[0]["content"]
    assert wired._processor.messages[0]["role"] == "user"
    assert content[0] == {"type": "image"}
    assert content[1]["type"] == "text"
    assert content[1]["text"] == load_perception_config()["prompt"]
    assert wired._processor.images == [picture]
    assert wired._processor.text == "PROMPT"


def test_generate_forwards_the_configured_generation_settings(wired) -> None:
    wired._generate(Image.new("RGB", (4, 4)))
    assert wired._model.kwargs["do_sample"] is False
    assert wired._model.kwargs["max_new_tokens"] == 160
    assert "input_ids" in wired._model.kwargs


def test_predict_parses_the_generated_text(wired) -> None:
    description = wired.predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert description.image_id == "in_memory"
    assert description.source_model == DEFAULT_MODEL_ID
    assert [garment.category for garment in description.garments] == ["tee"]
    assert description.garments[0].color is ColorName.white


def test_cpu_loads_in_float32(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    captured: dict = {}

    class Loaded:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_model(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["dtype"] = kwargs["dtype"]
        return Loaded()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda mid: object())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_model)

    model = SmolVLMPerception(device="cpu")
    model._ensure_loaded()
    assert captured["dtype"] is torch.float32
    assert captured["device"] == "cpu"
    assert captured["model_id"] == DEFAULT_MODEL_ID


def test_cuda_loads_in_float16(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    captured: dict = {}

    class Loaded:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_model(model_id, **kwargs):
        captured["dtype"] = kwargs["dtype"]
        return Loaded()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda mid: object())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_model)

    model = SmolVLMPerception(device="cuda")
    model._ensure_loaded()
    assert captured["dtype"] is torch.float16
    assert captured["device"] == "cuda"
