from __future__ import annotations

import pytest

from rmo.data.color_ground_truth_audit import (
    audit_color_descriptions,
    chroma,
    hue_degrees,
    hue_family,
)
from rmo.schemas import ColorName, Garment, GarmentSlot, OutfitDescription, Provenance
from rmo.scoring.palette import reference_lab


def _description(*garments: Garment) -> OutfitDescription:
    return OutfitDescription(
        image_id="sample",
        garments=list(garments),
        provenance=Provenance.gt,
        source_model="dataset_labels",
    )


def _garment(name: ColorName, lab: tuple[float, float, float]) -> Garment:
    return Garment(
        ref="upper_0",
        slot=GarmentSlot.upper,
        category="top",
        color=name,
        color_lab=lab,
        color_lab_source="mask",
    )


def test_chroma_uses_lab_ab_plane() -> None:
    assert chroma((50.0, 3.0, 4.0)) == pytest.approx(5.0)


def test_hue_degrees_wraps_to_positive_circle() -> None:
    assert hue_degrees((50.0, 1.0, 0.0)) == pytest.approx(0.0)
    assert hue_degrees((50.0, 0.0, -1.0)) == pytest.approx(270.0)


@pytest.mark.parametrize("name", [ColorName.red, ColorName.green, ColorName.blue, ColorName.rose])
def test_hue_family_recovers_reference_hue(name: ColorName) -> None:
    lab = reference_lab(name)
    assert lab is not None
    assert hue_family(lab) is name


def test_audit_flags_high_chroma_gray_assignment() -> None:
    blue_lab = reference_lab(ColorName.blue)
    assert blue_lab is not None
    report = audit_color_descriptions(
        [_description(_garment(ColorName.gray, blue_lab))],
        high_chroma_threshold=25.0,
    )
    assert report.garments_with_lab == 1
    assert report.achromatic_assignments == 1
    assert report.high_chroma_achromatic == 1
    assert report.achromatic_by_hue_family == {"blue": 1}
    assert report.suspicious_examples[0].assigned == "gray"
    assert report.suspicious_examples[0].hue_family == "blue"


def test_brown_is_not_treated_as_achromatic_for_this_diagnostic() -> None:
    brown_lab = reference_lab(ColorName.brown)
    assert brown_lab is not None
    report = audit_color_descriptions([_description(_garment(ColorName.brown, brown_lab))])
    assert report.achromatic_assignments == 0


def test_audit_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="high_chroma_threshold"):
        audit_color_descriptions([], high_chroma_threshold=0)
    with pytest.raises(ValueError, match="max_examples"):
        audit_color_descriptions([], max_examples=0)


def test_color_audit_cli_sample_selection_is_deterministic() -> None:
    # The CLI helper is deliberately imported from the script so sampling logic
    # stays testable without touching the raw image dataset.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "audit_smolvlm_colors.py"
    spec = importlib.util.spec_from_file_location("audit_smolvlm_colors_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = module.load_split
    try:
        module.load_split = lambda _name: {"a", "b", "c", "d", "e"}
        first = module._selected_ids("train", limit=None, sample=3, seed=7)
        second = module._selected_ids("train", limit=None, sample=3, seed=7)
        assert first == second
        assert len(first) == 3
        assert len(set(first)) == 3
    finally:
        module.load_split = original


def test_color_audit_cli_rejects_oversized_sample() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "audit_smolvlm_colors.py"
    spec = importlib.util.spec_from_file_location("audit_smolvlm_colors_script_oversized", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = module.load_split
    try:
        module.load_split = lambda _name: {"a", "b"}
        with pytest.raises(ValueError, match="exceeds split size"):
            module._selected_ids("train", limit=None, sample=3, seed=7)
    finally:
        module.load_split = original
