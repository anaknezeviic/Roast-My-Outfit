"""Cover the SmolVLM entry point without importing torch."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rmo.config import ConfigError, load_perception_config
from rmo.perception import vlm
from rmo.perception.base import PerceptionModel
from rmo.perception.vlm import (
    DEFAULT_MODEL_ID,
    _SLOT_PRIORITY,
    SmolVLMPerception,
    _apply_palette,
    _image_identity,
    _load_mask,
    _slot_labels,
)
from rmo.schemas import ColorName, Garment, GarmentSlot, OutfitDescription


def write_mask(root: Path, image_id: str, array: np.ndarray) -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(directory / f"{image_id}.png")


@pytest.fixture()
def two_regions() -> tuple[Image.Image, np.ndarray]:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[:7, :, 0] = 255
    pixels[7:, :, 2] = 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7] = 1
    mask[7:] = 5
    return Image.fromarray(pixels), mask


@pytest.fixture()
def labels() -> dict[GarmentSlot, frozenset[int]]:
    return _slot_labels(load_perception_config())


def test_module_imports_without_torch() -> None:
    assert importlib.import_module("rmo.perception.vlm") is vlm
    assert "torch" not in sys.modules


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


def test_slot_priority_covers_every_slot() -> None:
    assert set(_SLOT_PRIORITY) == set(GarmentSlot)
    assert len(_SLOT_PRIORITY) == len(GarmentSlot)


def test_image_identity_uses_the_filename_stem() -> None:
    stem = "WOMEN-Blouses-id_00000001-01_1_front"
    text = f"a/b/{stem}.jpg"
    assert _image_identity(text) == (stem, text)
    assert _image_identity(Path(text)) == (stem, str(Path(text)))


def test_image_identity_of_an_in_memory_image() -> None:
    assert _image_identity(np.zeros((2, 2, 3), dtype=np.uint8)) == ("in_memory", "")


def test_slot_labels_cover_every_slot(labels) -> None:
    assert set(labels) == set(GarmentSlot)
    assert labels[GarmentSlot.other] == frozenset()


def test_load_mask_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    assert _load_mask("missing", (10, 10), frozenset({1})) is None


def test_load_mask_rejects_a_shape_mismatch(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    write_mask(tmp_path, "small", np.ones((4, 4), dtype=np.uint8))
    with caplog.at_level(logging.WARNING, logger="rmo.perception.vlm"):
        assert _load_mask("small", (10, 10), frozenset({1})) is None
    assert len(caplog.records) == 1


def test_load_mask_zeroes_unclaimed_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.zeros((6, 6), dtype=np.uint8)
    array[:2] = 1
    array[2:4] = 13
    array[4:] = 15
    write_mask(tmp_path, "mixed", array)
    mask = _load_mask("mixed", (6, 6), frozenset({1}))
    assert mask is not None
    assert set(np.unique(mask).tolist()) == {0, 1}


def test_load_mask_returns_none_when_nothing_is_claimed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.full((6, 6), 13, dtype=np.uint8)
    array[3:] = 15
    write_mask(tmp_path, "skin", array)
    assert _load_mask("skin", (6, 6), frozenset({1})) is None


def test_load_mask_takes_the_first_channel_of_an_rgb_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    array = np.zeros((6, 6, 3), dtype=np.uint8)
    array[..., 0] = 2
    write_mask(tmp_path, "rgb", array)
    mask = _load_mask("rgb", (6, 6), frozenset({2}))
    assert mask is not None
    assert mask.ndim == 2
    assert set(np.unique(mask).tolist()) == {2}


def test_apply_palette_joins_mask_regions_to_slots(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.lower, category="jeans"),
    ]
    _apply_palette(garments, image, mask, labels)
    assert [garment.color for garment in garments] == [ColorName.red, ColorName.blue]
    assert all(garment.color_lab_source == "mask" for garment in garments)
    assert [garment.area_fraction for garment in garments] == pytest.approx([0.7, 0.3])


def test_apply_palette_area_fractions_sum_to_one(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.lower, category="jeans"),
    ]
    _apply_palette(garments, image, mask, labels)
    total = sum(garment.area_fraction for garment in garments)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_apply_palette_skips_slots_with_no_mask_pixels(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.footwear, category="boots"),
    ]
    _apply_palette(garments, image, mask, labels)
    boots = garments[1]
    assert boots.color_lab is None
    assert boots.color_lab_source is None
    assert boots.area_fraction is None


def test_apply_palette_shares_one_region_between_two_uppers(two_regions, labels) -> None:
    image, mask = two_regions
    description = OutfitDescription(
        image_id="fixture_000",
        source_model="fixture",
        garments=[
            Garment(slot=GarmentSlot.upper, category="tee"),
            Garment(slot=GarmentSlot.upper, category="cardigan"),
        ],
    )
    _apply_palette(description.garments, image, mask, labels)
    first, second = description.garments
    assert first.color_lab == second.color_lab
    assert description.refs() == ["upper_0", "upper_1"]


def test_apply_palette_keeps_a_parsed_colour(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.lower, category="jeans", color=ColorName.red)]
    _apply_palette(garments, image, mask, labels)
    assert garments[0].color is ColorName.red
    assert garments[0].color_lab is not None


def test_apply_palette_fills_only_unknown_colours(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.lower, category="jeans")]
    _apply_palette(garments, image, mask, labels)
    assert garments[0].color is ColorName.blue


def test_apply_palette_without_a_mask_colours_the_priority_garment(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [
        Garment(slot=GarmentSlot.footwear, category="boots"),
        Garment(slot=GarmentSlot.dress, category="gown"),
    ]
    _apply_palette(garments, image, None, labels)
    assert garments[1].color_lab_source == "wholeimage"
    assert garments[0].color_lab_source is None
    assert garments[0].color_lab is None


def test_apply_palette_without_a_mask_leaves_area_fraction_none(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [Garment(slot=GarmentSlot.dress, category="gown")]
    _apply_palette(garments, image, None, labels)
    assert garments[0].area_fraction is None


def test_apply_palette_breaks_priority_ties_by_order(two_regions, labels) -> None:
    image, _ = two_regions
    garments = [
        Garment(slot=GarmentSlot.upper, category="tee"),
        Garment(slot=GarmentSlot.upper, category="cardigan"),
    ]
    _apply_palette(garments, image, None, labels)
    assert garments[0].color_lab is not None
    assert garments[1].color_lab is None


def test_apply_palette_on_a_fallback_description_does_not_raise(two_regions, labels) -> None:
    image, mask = two_regions
    garments = [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)]
    _apply_palette(garments, image, mask, labels)
    assert garments[0].color_lab is None
    assert garments[0].color is ColorName.unknown


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
