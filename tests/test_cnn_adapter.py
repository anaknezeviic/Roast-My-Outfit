"""Cover the CNN perception adapter without loading real weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rmo import paths, pipeline
from rmo.imaging import load_image
from rmo.perception.base import PerceptionModel
from rmo.perception.cnn_adapter import CNNPerception
from rmo.perception.enrichment import HEAD_PROJECTION
from rmo.schemas import (
    Fabric,
    Garment,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
)

IMPORT_PROBE = (
    "import sys; import rmo.perception; "
    "from rmo.pipeline import registered_names; registered_names(); "
    "print('torch' in sys.modules, 'timm' in sys.modules)"
)

LABELS = {
    "upper_fabric": "denim",
    "lower_fabric": "cotton",
    "outer_fabric": "leather",
    "upper_pattern": "striped",
    "lower_pattern": "pure_color",
    "outer_pattern": "floral",
    "sleeve_length": "long",
    "lower_length": "three_quarter",
    "neckline": "v_shape",
}

BOOSTS = {head: 1.0 + index for index, head in enumerate(LABELS)}


@pytest.fixture(scope="module")
def cnn():
    return pytest.importorskip("rmo.perception.cnn")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


class StubModel:
    """Return one-hot logits whose winner and margin are fixed per head."""

    def __init__(self, torch, cnn, labels, *, boosts=None, infinite=()) -> None:
        self._torch = torch
        self._cnn = cnn
        self.labels = dict(labels)
        self.boosts = dict(BOOSTS if boosts is None else boosts)
        self.infinite = frozenset(infinite)
        self.batches: list[int] = []

    def __call__(self, pixels):
        self.batches.append(int(pixels.shape[0]))
        logits = {}
        for head, size in self._cnn.HEAD_CLASS_COUNTS.items():
            values = self._torch.zeros(int(pixels.shape[0]), size)
            index = self._cnn.CLASS_MAPS[head].index(self.labels[head])
            values[:, index] = float("inf") if head in self.infinite else self.boosts[head]
            logits[head] = values
        return logits


def wire(model, torch, cnn, **options) -> StubModel:
    stub = StubModel(torch, cnn, options.pop("labels", LABELS), **options)
    model._torch = torch
    model._cnn = cnn
    model._device = "cpu"
    model._transform = cnn.build_transform(model._config)
    model._model = stub
    return stub


def write_mask(root: Path, image_id: str, array: np.ndarray) -> None:
    directory = root / "raw" / "parsing"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(directory / f"{image_id}_segm.png")


def banded_image(path: Path) -> Path:
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[:7, :, 0] = 255
    pixels[7:, :, 2] = 255
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))

    def stage(image_id: str, mask: np.ndarray | None = None) -> Path:
        photo = banded_image(tmp_path / f"{image_id}.png")
        if mask is not None:
            write_mask(tmp_path, image_id, mask)
        return photo

    return stage


def upper_lower_mask() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7] = 1
    mask[7:] = 5
    return mask


def cornered_mask(*, skin: bool) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7, :7] = 1
    mask[7:, :7] = 5
    if skin:
        mask[:, 7:] = 15
    return mask


def expected_confidence(readout, heads) -> float:
    return sum(readout.probabilities[head] for head in heads) / len(heads)


def test_importing_the_package_leaves_torch_and_timm_alone() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "False False"


def test_the_adapter_is_registered() -> None:
    assert CNNPerception.name == "cnn_multihead_v1"
    assert CNNPerception.name in pipeline.registered_names()


def test_the_adapter_is_a_perception_model() -> None:
    assert issubclass(CNNPerception, PerceptionModel)


def test_the_constructor_loads_no_weights() -> None:
    model = CNNPerception()
    assert model._model is None
    assert model._checkpoint_path.is_absolute()
    assert model._batch_size == 16


def test_head_projection_covers_exactly_the_nine_heads(cnn) -> None:
    assert tuple(HEAD_PROJECTION) == cnn.HEAD_NAMES


@pytest.mark.parametrize("head", list(HEAD_PROJECTION))
def test_head_projection_targets_a_real_garment_field(head) -> None:
    field, candidates = HEAD_PROJECTION[head]
    assert field in Garment.model_fields
    assert candidates and set(candidates) <= set(GarmentSlot)


def test_the_shape_heads_reach_the_same_slots_as_the_label_arm() -> None:
    assert HEAD_PROJECTION["sleeve_length"][1] == (GarmentSlot.upper, GarmentSlot.outer)
    assert HEAD_PROJECTION["neckline"][1] == HEAD_PROJECTION["sleeve_length"][1]
    assert HEAD_PROJECTION["lower_length"][1] == (GarmentSlot.lower,)


def test_no_head_reaches_a_dress_or_a_romper() -> None:
    reached = {
        slot for _, candidates in HEAD_PROJECTION.values() for slot in candidates
    }
    assert reached.isdisjoint({GarmentSlot.dress, GarmentSlot.romper})


def test_a_prediction_round_trips_through_the_schema(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    description = model.predict(photo)
    assert description.source_model == "cnn_multihead_v1"
    assert OutfitDescription.model_validate(description.model_dump(mode="json")) == description


def test_one_garment_per_populated_slot(staged, torch, cnn) -> None:
    mask = upper_lower_mask()
    mask[8:] = 11
    photo = staged("outfit", mask)
    model = CNNPerception()
    wire(model, torch, cnn)
    description = model.predict(photo)
    assert description.refs() == ["upper_0", "lower_0", "footwear_0"]
    assert [garment.slot for garment in description.garments] == [
        GarmentSlot.upper,
        GarmentSlot.lower,
        GarmentSlot.footwear,
    ]


def test_the_upper_heads_land_only_on_the_upper_garment(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    upper, lower = model.predict(photo).garments
    assert (upper.fabric, upper.pattern) == (Fabric.denim, Pattern.striped)
    assert (lower.fabric, lower.pattern) == (Fabric.cotton, Pattern.pure_color)


def test_the_lower_length_head_writes_the_length_field(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    upper, lower = model.predict(photo).garments
    assert lower.length is LowerLength.three_quarter
    assert upper.length is None


def test_sleeve_and_neckline_prefer_the_upper_garment(staged, torch, cnn) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 1
    mask[5:] = 2
    photo = staged("outfit", mask)
    model = CNNPerception()
    wire(model, torch, cnn)
    upper, outer = model.predict(photo).garments
    assert (upper.sleeve_length.value, upper.neckline is Neckline.v_shape) == ("long", True)
    assert outer.sleeve_length is None
    assert outer.neckline is None


def test_sleeve_and_neckline_fall_back_to_the_outer_garment(staged, torch, cnn) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:] = 2
    photo = staged("outfit", mask)
    model = CNNPerception()
    wire(model, torch, cnn)
    (outer,) = model.predict(photo).garments
    assert outer.slot is GarmentSlot.outer
    assert outer.sleeve_length.value == "long"
    assert outer.neckline is Neckline.v_shape


def test_a_dress_stays_unmapped(staged, torch, cnn) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:7] = 4
    mask[7:] = 11
    photo = staged("outfit", mask)
    model = CNNPerception()
    wire(model, torch, cnn)
    description, readout = model.predict_with_readout(photo)
    dress = description.garments[0]
    assert dress.slot is GarmentSlot.dress
    assert (dress.fabric, dress.pattern) == (Fabric.na, Pattern.na)
    assert (dress.sleeve_length, dress.length, dress.neckline) == (None, None, None)
    assert dress.confidence == 0.0
    assert readout.unmapped_refs == ("dress_0", "footwear_0")


def test_confidence_is_the_mean_of_the_assigned_heads(staged, torch, cnn) -> None:
    mask = upper_lower_mask()
    mask[8:] = 11
    photo = staged("outfit", mask)
    model = CNNPerception()
    wire(model, torch, cnn)
    description, readout = model.predict_with_readout(photo)
    upper, lower, footwear = description.garments
    assert upper.confidence == pytest.approx(
        expected_confidence(readout, ["upper_fabric", "upper_pattern", "sleeve_length", "neckline"]),
        abs=1e-6,
    )
    assert lower.confidence == pytest.approx(
        expected_confidence(readout, ["lower_fabric", "lower_pattern", "lower_length"]), abs=1e-6
    )
    assert footwear.confidence == 0.0


def test_the_readout_records_the_dominant_mask_class(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    _, readout = model.predict_with_readout(photo)
    assert readout.mask_classes == {"upper_0": 1, "lower_0": 5}
    assert readout.fallback is False
    assert readout.labels == LABELS


def test_the_mask_fills_colour_and_area(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    upper, lower = model.predict(photo).garments
    assert [garment.color.value for garment in (upper, lower)] == ["red", "blue"]
    assert all(garment.color_lab_source == "mask" for garment in (upper, lower))
    assert [upper.area_fraction, lower.area_fraction] == pytest.approx([0.7, 0.3])


def test_unclaimed_pixels_stay_out_of_the_area_denominator(staged, torch, cnn) -> None:
    plain = staged("plain", cornered_mask(skin=False))
    skinned = staged("skinned", cornered_mask(skin=True))
    model = CNNPerception()
    wire(model, torch, cnn)
    without = [garment.area_fraction for garment in model.predict(plain).garments]
    with_skin = [garment.area_fraction for garment in model.predict(skinned).garments]
    assert without == pytest.approx([0.7, 0.3])
    assert with_skin == pytest.approx(without)


def test_a_missing_mask_falls_back_to_one_unlocated_garment(staged, torch, cnn) -> None:
    photo = staged("outfit")
    model = CNNPerception()
    wire(model, torch, cnn)
    description, readout = model.predict_with_readout(photo)
    (garment,) = description.garments
    assert (garment.slot, garment.category) == (GarmentSlot.other, "unknown")
    assert (garment.fabric, garment.pattern) == (Fabric.denim, Pattern.striped)
    assert (garment.sleeve_length, garment.length, garment.neckline) == (None, None, None)
    assert garment.area_fraction is None
    assert garment.color_lab_source == "wholeimage"
    assert readout.fallback is True
    assert readout.mask_classes == {}
    assert garment.confidence == pytest.approx(
        expected_confidence(readout, ["upper_fabric", "upper_pattern"]), abs=1e-6
    )


def test_the_fallback_takes_the_first_texture_that_is_not_na(staged, torch, cnn) -> None:
    photo = staged("outfit")
    labels = {**LABELS, "upper_fabric": "na", "upper_pattern": "na", "lower_pattern": "na"}
    model = CNNPerception()
    wire(model, torch, cnn, labels=labels)
    description, readout = model.predict_with_readout(photo)
    (garment,) = description.garments
    assert garment.fabric is Fabric.cotton
    assert garment.pattern is Pattern.floral
    assert garment.confidence == pytest.approx(
        expected_confidence(readout, ["lower_fabric", "outer_pattern"]), abs=1e-6
    )


def test_an_all_na_fallback_is_still_a_valid_description(staged, torch, cnn) -> None:
    photo = staged("outfit")
    labels = dict.fromkeys(LABELS, "na")
    model = CNNPerception()
    wire(model, torch, cnn, labels=labels)
    description = model.predict(photo)
    (garment,) = description.garments
    assert (garment.fabric, garment.pattern) == (Fabric.na, Pattern.na)
    assert garment.confidence == 0.0
    assert OutfitDescription.model_validate(description.model_dump(mode="json")) == description


@pytest.mark.parametrize("kind", ["str", "path", "pil", "array"])
def test_every_input_type_is_accepted(staged, torch, cnn, kind) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn)
    image = {
        "str": str(photo),
        "path": photo,
        "pil": Image.open(photo),
        "array": np.asarray(load_image(photo)),
    }[kind]
    description = model.predict(image)
    assert description.image_id == ("in_memory" if kind == "array" else "outfit")
    assert description.garments


def test_the_image_path_is_repo_relative(staged, torch, cnn) -> None:
    model = CNNPerception()
    wire(model, torch, cnn)
    inside = paths.repo_root() / "data" / "fixtures" / "images" / "fixture_000.png"
    assert model.predict(inside).image_path == "data/fixtures/images/fixture_000.png"


def test_an_in_memory_image_has_no_path(staged, torch, cnn) -> None:
    model = CNNPerception()
    wire(model, torch, cnn)
    assert model.predict(np.zeros((6, 6, 3), dtype=np.uint8)).image_path == ""


def test_a_batch_keeps_its_input_order(staged, torch, cnn) -> None:
    photos = [staged(f"outfit_{index}", upper_lower_mask()) for index in range(5)]
    model = CNNPerception()
    stub = wire(model, torch, cnn)
    model._batch_size = 2
    descriptions = model.predict_batch(list(reversed(photos)))
    assert [description.image_id for description in descriptions] == [
        f"outfit_{index}" for index in reversed(range(5))
    ]
    assert stub.batches == [2, 2, 1]


def test_an_empty_batch_returns_nothing() -> None:
    model = CNNPerception()
    assert model.predict_batch([]) == []
    assert model.predict_batch_with_readouts([]) == []
    assert model._model is None


def test_a_batch_agrees_with_single_inference(staged, torch, cnn) -> None:
    photos = [staged(f"outfit_{index}", upper_lower_mask()) for index in range(3)]
    model = CNNPerception()
    wire(model, torch, cnn)
    model._batch_size = 2
    batched = model.predict_batch(photos)
    singles = [model.predict(photo) for photo in photos]
    assert [item.model_dump() for item in batched] == [item.model_dump() for item in singles]


def test_a_non_finite_logit_names_its_head(staged, torch, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception()
    wire(model, torch, cnn, infinite={"neckline"})
    with pytest.raises(ValueError, match="neckline"):
        model.predict(photo)


@pytest.fixture()
def checkpoint(tmp_path, monkeypatch, torch, cnn):
    monkeypatch.setattr(
        cnn,
        "build_backbone",
        lambda name, **options: (
            torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)),
            4,
        ),
    )
    destination = tmp_path / "cnn.pt"
    cnn.save_checkpoint(
        destination,
        cnn.MultiHeadCNN(
            torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
            ),
            4,
        ),
        backbone="tiny",
        preprocessing={**cnn.DEFAULT_PREPROCESSING, "image_size": 64},
    )
    return destination


def test_a_preprocessing_mismatch_is_reported(staged, checkpoint, cnn) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception(checkpoint_path=checkpoint)
    with pytest.raises(cnn.CheckpointError, match="preprocessing"):
        model.predict(photo)


def test_an_absent_checkpoint_is_reported(staged, tmp_path) -> None:
    photo = staged("outfit", upper_lower_mask())
    model = CNNPerception(checkpoint_path=tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="absent.pt"):
        model.predict(photo)
