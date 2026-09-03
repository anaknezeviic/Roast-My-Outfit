"""Check perception metrics on tiny hand-computed examples."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from rmo.eval.perception_eval import (
    compute_metrics,
    evaluate_predictions,
    is_fallback,
    log_evaluation,
)
from rmo.schemas import (
    Fabric,
    Garment,
    GarmentSlot,
    LowerLength,
    Neckline,
    OutfitDescription,
    Pattern,
    SleeveLength,
)


def description(image_id: str, garments: list[Garment]) -> OutfitDescription:
    """Build a compact prediction for one synthetic image."""
    return OutfitDescription(image_id=image_id, source_model="test", garments=garments)


def test_compute_metrics_includes_na_in_every_metric() -> None:
    metrics = compute_metrics(
        ["cotton", "cotton", "denim", "na"],
        ["cotton", "denim", "denim", "cotton"],
    )
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(7 / 18)
    assert metrics.sample_size == 4
    assert metrics.confusion == {
        ("cotton", "cotton"): 1,
        ("cotton", "denim"): 1,
        ("denim", "denim"): 1,
        ("na", "cotton"): 1,
    }


def test_compute_metrics_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(["cotton"], [])


def test_fallback_detection_matches_the_parser_sentinel() -> None:
    fallback = description(
        "fallback",
        [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)],
    )
    ordinary = description(
        "ordinary",
        [Garment(slot=GarmentSlot.upper, category="unknown", confidence=0.0)],
    )
    assert is_fallback(fallback)
    assert not is_fallback(ordinary)


def test_evaluate_predictions_aligns_slots_and_counts_missing_as_na() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "one",
                "upper_fabric": "cotton",
                "lower_fabric": "denim",
                "outer_fabric": "na",
                "upper_pattern": "striped",
                "lower_pattern": "pure_color",
                "outer_pattern": "na",
                "sleeve_length": "short",
                "lower_length": "long",
                "neckline": "round",
            },
            {
                "image_id": "two",
                "upper_fabric": "na",
                "lower_fabric": "na",
                "outer_fabric": "na",
                "upper_pattern": "na",
                "lower_pattern": "na",
                "outer_pattern": "na",
                "sleeve_length": "na",
                "lower_length": "na",
                "neckline": "na",
            },
        ]
    )
    predictions = [
        description(
            "one",
            [
                Garment(
                    slot=GarmentSlot.upper,
                    category="shirt",
                    fabric=Fabric.cotton,
                    pattern=Pattern.striped,
                    sleeve_length=SleeveLength.short,
                    neckline=Neckline.round,
                ),
                Garment(
                    slot=GarmentSlot.lower,
                    category="jeans",
                    fabric=Fabric.denim,
                    pattern=Pattern.pure_color,
                    length=LowerLength.long,
                ),
            ],
        ),
        description(
            "two",
            [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)],
        ),
    ]

    result = evaluate_predictions(frame, predictions, {"one", "two"})

    assert set(result.fields) == {
        "fabric",
        "pattern",
        "sleeve_length",
        "length",
        "neckline",
    }
    assert result.fields["fabric"].sample_size == 6
    assert result.fields["fabric"].accuracy == 1.0
    assert result.fields["sleeve_length"].sample_size == 2
    assert result.fields["sleeve_length"].confusion["na", "na"] == 1
    assert result.schema_validity == 0.5
    assert result.sample_size == 2


def test_evaluate_predictions_requires_exact_test_coverage() -> None:
    frame = pd.DataFrame({"image_id": ["one"]})
    with pytest.raises(ValueError, match="labels do not exactly match"):
        evaluate_predictions(frame, [], {"one", "two"})


def test_shape_metrics_use_the_upper_garment_under_outerwear() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "layered",
                "upper_fabric": "na",
                "lower_fabric": "na",
                "outer_fabric": "na",
                "upper_pattern": "na",
                "lower_pattern": "na",
                "outer_pattern": "na",
                "sleeve_length": "short",
                "lower_length": "na",
                "neckline": "round",
            }
        ]
    )
    prediction = description(
        "layered",
        [
            Garment(
                slot=GarmentSlot.upper,
                category="shirt",
                sleeve_length=SleeveLength.short,
                neckline=Neckline.round,
            ),
            Garment(
                slot=GarmentSlot.outer,
                category="coat",
                sleeve_length=SleeveLength.long,
                neckline=Neckline.lapel,
            ),
        ],
    )

    result = evaluate_predictions(frame, [prediction], {"layered"})

    assert result.fields["sleeve_length"].accuracy == 1.0
    assert result.fields["neckline"].accuracy == 1.0


def test_texture_metrics_map_dress_to_both_dataset_regions() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "dress",
                "upper_fabric": "chiffon",
                "lower_fabric": "chiffon",
                "outer_fabric": "na",
                "upper_pattern": "floral",
                "lower_pattern": "floral",
                "outer_pattern": "na",
                "sleeve_length": "na",
                "lower_length": "na",
                "neckline": "na",
            }
        ]
    )
    prediction = description(
        "dress",
        [
            Garment(
                slot=GarmentSlot.dress,
                category="dress",
                fabric=Fabric.chiffon,
                pattern=Pattern.floral,
            )
        ],
    )

    result = evaluate_predictions(frame, [prediction], {"dress"})

    assert result.fields["fabric"].accuracy == 1.0
    assert result.fields["pattern"].accuracy == 1.0


def test_texture_metrics_map_romper_to_both_dataset_regions() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "romper",
                "upper_fabric": "denim",
                "lower_fabric": "denim",
                "outer_fabric": "na",
                "upper_pattern": "pure_color",
                "lower_pattern": "pure_color",
                "outer_pattern": "na",
                "sleeve_length": "na",
                "lower_length": "na",
                "neckline": "na",
            }
        ]
    )
    prediction = description(
        "romper",
        [
            Garment(
                slot=GarmentSlot.romper,
                category="romper",
                fabric=Fabric.denim,
                pattern=Pattern.pure_color,
            )
        ],
    )

    result = evaluate_predictions(frame, [prediction], {"romper"})

    assert result.fields["fabric"].accuracy == 1.0
    assert result.fields["pattern"].accuracy == 1.0


def test_log_evaluation_reports_fields_confusions_and_sample_size(caplog) -> None:
    metrics = compute_metrics(["na"], ["na"])
    from rmo.eval.perception_eval import EvaluationResult

    result = EvaluationResult(
        fields={"fabric": metrics},
        schema_validity=1.0,
        valid_generations=1,
        sample_size=1,
    )
    with caplog.at_level(logging.INFO, logger="rmo.eval.perception_eval"):
        log_evaluation(result)
    output = "\n".join(record.message for record in caplog.records)
    assert "field=fabric accuracy=1.000000 macro_f1=1.000000" in output
    assert "field=fabric actual=na predicted=na count=1" in output
    assert "schema_validity=1.000000 valid=1" in output