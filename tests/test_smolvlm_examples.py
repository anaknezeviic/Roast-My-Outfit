"""SmolVLM supervision must be a lossless view of canonical ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from rmo.config import load_perception_config
from rmo.data.smolvlm_examples import (
    SMOLVLM_PROMPT_V1,
    build_training_example,
    smolvlm_prompt,
    training_omissions,
)
from rmo.schemas import Garment, GarmentSlot, OutfitDescription

IMAGE_ID = "WOMEN-Tees_Tanks-id_00000001-01_7_additional"

DEFAULTS = {
    "upper_fabric": "na",
    "upper_pattern": "na",
    "outer_fabric": "na",
    "outer_pattern": "na",
    "lower_fabric": "na",
    "lower_pattern": "na",
    "sleeve_length": "na",
    "lower_length": "na",
    "neckline": "na",
    "has_shape": False,
    "caption": "",
    "category_from_filename": "Tees_Tanks",
}


def row(**overrides: object) -> pd.Series:
    return pd.Series({**DEFAULTS, **overrides})


def stage(root: Path, mask: np.ndarray) -> None:
    images = root / "raw" / "images"
    parsing = root / "raw" / "parsing"
    images.mkdir(parents=True, exist_ok=True)
    parsing.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((*mask.shape, 3), dtype=np.uint8)
    pixels[..., 2] = 255
    Image.fromarray(pixels).save(images / f"{IMAGE_ID}.jpg", quality=100)
    Image.fromarray(mask.astype(np.uint8)).save(parsing / f"{IMAGE_ID}_segm.png")


def test_default_training_prompt_is_the_runtime_perception_prompt() -> None:
    expected = load_perception_config()["prompt"].strip()
    assert SMOLVLM_PROMPT_V1 == expected
    assert smolvlm_prompt() == expected


def test_builder_uses_mask_presence_even_when_texture_is_na(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    mask = np.ones((8, 8), dtype=np.uint8)  # upper parsing class
    stage(tmp_path, mask)
    example = build_training_example(IMAGE_ID, row())
    assert example.image_id == IMAGE_ID
    assert example.image_path.is_file()
    assert example.prompt == SMOLVLM_PROMPT_V1
    assert example.answer.startswith("upper: top | color=blue")
    assert "pattern=na" in example.answer
    assert "fabric=na" in example.answer


def test_product_filename_category_is_not_spread_over_visible_slots(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4] = 1   # upper
    mask[4:] = 3   # lower
    stage(tmp_path, mask)
    example = build_training_example(
        IMAGE_ID,
        row(category_from_filename="Blouses_Shirts", lower_fabric="denim"),
    )
    lines = example.answer.splitlines()
    assert lines[0].startswith("upper: top |")
    assert lines[1].startswith("lower: bottom |")
    assert "Blouses_Shirts" not in example.answer


def test_accessory_texture_defaults_are_not_supervised(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage(tmp_path, np.full((8, 8), 11, dtype=np.uint8))  # footwear
    example = build_training_example(IMAGE_ID, row())
    assert example.answer.startswith("footwear: footwear | color=blue")
    assert "fabric=" not in example.answer
    assert "pattern=" not in example.answer


def test_one_piece_conflicts_are_omitted_not_supervised_as_na(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage(tmp_path, np.full((8, 8), 4, dtype=np.uint8))  # dress
    example = build_training_example(
        IMAGE_ID,
        row(
            upper_fabric="cotton",
            lower_fabric="denim",
            upper_pattern="floral",
            lower_pattern="striped",
            has_shape=True,
            sleeve_length="long",
            lower_length="long",
            neckline="round",
        ),
    )
    assert example.answer.startswith("dress: dress | color=blue")
    assert "fabric=" not in example.answer
    assert "pattern=" not in example.answer
    assert "sleeve_length=long" in example.answer
    assert "length=long" in example.answer
    assert "neckline=round" in example.answer


def test_one_piece_equal_labels_are_supervised() -> None:
    description = OutfitDescription(
        image_id="dress",
        source_model="gt",
        garments=[Garment(ref="dress_0", slot=GarmentSlot.dress, category="dress")],
    )
    omissions = training_omissions(
        description,
        {
            "upper_fabric": "chiffon",
            "lower_fabric": "chiffon",
            "upper_pattern": "floral",
            "lower_pattern": "floral",
        },
    )
    assert omissions == {}


def test_hf_record_has_exactly_the_vlm_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    stage(tmp_path, np.ones((8, 8), dtype=np.uint8))
    record = build_training_example(IMAGE_ID, row()).as_hf_record()
    assert set(record) == {"image", "prompt", "completion"}
    assert record["prompt"][0]["role"] == "user"
    assert record["prompt"][0]["content"][0] == {"type": "image"}
    assert record["completion"][0]["role"] == "assistant"
