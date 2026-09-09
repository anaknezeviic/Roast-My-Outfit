"""Canonical regional-label resolution must never invent one-piece labels."""

from __future__ import annotations

import pytest

from rmo.data.label_resolution import resolve_regional_value, texture_value
from rmo.schemas import GarmentSlot


@pytest.mark.parametrize(
    ("upper", "lower", "value", "kind"),
    [
        ("na", "na", "na", "both_na"),
        ("cotton", "na", "cotton", "upper_only"),
        ("na", "denim", "denim", "lower_only"),
        ("chiffon", "chiffon", "chiffon", "same"),
        ("cotton", "denim", "na", "conflict"),
    ],
)
def test_regional_resolution_is_deterministic(upper, lower, value, kind) -> None:
    resolution = resolve_regional_value(upper, lower)
    assert resolution.value == value
    assert resolution.kind == kind
    assert resolution.conflicted is (kind == "conflict")


def test_one_piece_texture_uses_the_same_rule_for_dress_and_romper() -> None:
    row = {
        "upper_fabric": "cotton",
        "lower_fabric": "na",
        "upper_pattern": "striped",
        "lower_pattern": "pure_color",
    }
    for slot in (GarmentSlot.dress, GarmentSlot.romper):
        assert texture_value(row, slot, "fabric") == "cotton"
        assert texture_value(row, slot, "pattern") == "na"


def test_regional_slots_keep_their_own_columns() -> None:
    row = {
        "upper_fabric": "cotton",
        "upper_pattern": "striped",
        "lower_fabric": "denim",
        "lower_pattern": "pure_color",
        "outer_fabric": "leather",
        "outer_pattern": "other",
    }
    assert texture_value(row, GarmentSlot.upper, "fabric") == "cotton"
    assert texture_value(row, GarmentSlot.lower, "pattern") == "pure_color"
    assert texture_value(row, GarmentSlot.outer, "fabric") == "leather"


def test_accessories_have_no_dataset_texture_supervision() -> None:
    assert texture_value({}, GarmentSlot.footwear, "fabric") == "na"
    assert texture_value({}, GarmentSlot.bag, "pattern") == "na"


def test_unknown_texture_attribute_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported texture attribute"):
        texture_value({}, GarmentSlot.upper, "colour")
