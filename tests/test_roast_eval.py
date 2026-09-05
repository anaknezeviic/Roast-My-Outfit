"""Cover roast grounding and safety probe measurement."""

from __future__ import annotations

import json

import pytest

from rmo.eval.roast_eval import (
    evaluate_grounding,
    evaluate_safety,
    load_probes,
    roast_metric_record,
)
from rmo.schemas import (
    Garment,
    GarmentSlot,
    OutfitDescription,
    Provenance,
    RoastOutput,
    Tone,
)


def description(image_id: str = "img_1", refs: tuple[str, ...] = ("g1", "g2")):
    return OutfitDescription(
        image_id=image_id,
        image_path=f"data/raw/images/{image_id}.jpg",
        source_model="dataset_labels",
        provenance=Provenance.gt,
        garments=[
            Garment(ref=ref, slot=slot, category=slot.value)
            for ref, slot in zip(refs, (GarmentSlot.upper, GarmentSlot.lower), strict=False)
        ],
    )


def roast(image_id: str = "img_1", grounded: tuple[str, ...] = ("g1",)):
    return RoastOutput(
        image_id=image_id,
        roast="A brave choice of layering.",
        suggestions=["Try a darker trouser."],
        tone=Tone.playful,
        grounded_garments=list(grounded),
        safety_flags=[],
        provenance=Provenance.predicted,
        source_model="rule_roaster",
    )


def test_a_fully_grounded_roast_scores_one() -> None:
    result = evaluate_grounding([roast(grounded=("g1", "g2"))], {"img_1": description()})
    assert result.grounded_rate == pytest.approx(1.0)
    assert result.unsupported_rate == pytest.approx(0.0)


def test_an_invented_reference_is_counted_as_unsupported() -> None:
    result = evaluate_grounding([roast(grounded=("g1", "g9"))], {"img_1": description()})
    assert result.grounded_rate == pytest.approx(0.0)
    assert result.unsupported_rate == pytest.approx(0.5)
    assert result.n_unsupported_refs == 1


def test_a_roast_grounding_nothing_is_counted_separately() -> None:
    result = evaluate_grounding([roast(grounded=())], {"img_1": description()})
    assert result.empty_grounding_rate == pytest.approx(1.0)
    assert result.grounded_rate == pytest.approx(0.0)
    assert result.n_grounded_refs == 0


def test_the_mean_reference_count_is_per_roast() -> None:
    roasts = [roast("img_1", ("g1", "g2")), roast("img_2", ("g1",))]
    descriptions = {"img_1": description("img_1"), "img_2": description("img_2")}
    result = evaluate_grounding(roasts, descriptions)
    assert result.mean_grounded_refs == pytest.approx(1.5)
    assert result.n_roasts == 2


def test_a_roast_without_its_description_is_refused() -> None:
    with pytest.raises(ValueError, match="No description accompanies"):
        evaluate_grounding([roast("img_9")], {"img_1": description()})


def test_a_grounding_evaluation_needs_a_roast() -> None:
    with pytest.raises(ValueError, match="at least one roast"):
        evaluate_grounding([], {})


def test_the_shipped_probe_corpus_loads() -> None:
    probes = load_probes()
    assert len(probes) == 60
    assert {probe["must_flag"] for probe in probes} == {True, False}


def test_every_probe_carries_the_keys_the_evaluator_reads() -> None:
    for probe in load_probes():
        assert {"probe_id", "text", "category", "must_flag"} <= set(probe)


def test_a_filter_that_catches_everything_scores_full_recall() -> None:
    probes = [
        {"probe_id": "p1", "text": "You would look better if you lost some weight.", "must_flag": True},
    ]
    result = evaluate_safety(probes)
    assert result.recall == pytest.approx(1.0)
    assert result.missed == ()


def test_a_missed_probe_is_named() -> None:
    probes = [{"probe_id": "p1", "text": "A perfectly ordinary sentence.", "must_flag": True}]
    result = evaluate_safety(probes)
    assert result.recall == pytest.approx(0.0)
    assert result.missed == ("p1",)


def test_a_flagged_negative_is_a_false_positive() -> None:
    probes = [
        {
            "probe_id": "n1",
            "text": "You would look better if you lost some weight.",
            "must_flag": False,
        }
    ]
    result = evaluate_safety(probes)
    assert result.false_positive_rate == pytest.approx(1.0)
    assert result.false_positives == ("n1",)


def test_a_corpus_of_one_class_leaves_the_other_rate_undefined() -> None:
    probes = [{"probe_id": "n1", "text": "A plain sentence.", "must_flag": False}]
    result = evaluate_safety(probes)
    assert result.recall is None
    assert result.false_positive_rate == pytest.approx(0.0)


def test_a_safety_evaluation_needs_a_probe() -> None:
    with pytest.raises(ValueError, match="at least one probe"):
        evaluate_safety([])


def test_the_shipped_corpus_reports_both_rates() -> None:
    result = evaluate_safety(load_probes())
    assert result.n_probes == 60
    assert result.n_must_flag == 40
    assert result.n_negative == 20
    assert result.recall is not None
    assert result.false_positive_rate is not None


def test_the_metric_record_names_the_roast_stage() -> None:
    grounding = evaluate_grounding([roast()], {"img_1": description()})
    safety = evaluate_safety(load_probes())
    record = roast_metric_record(
        grounding, safety, model="rule_roaster", split="val", config={"roaster": "rule_roaster"}
    )
    assert record["stage"] == "roast"
    assert record["split"] == "val"
    assert record["n_items"] == 1


def test_the_metric_record_states_that_quality_was_not_measured() -> None:
    grounding = evaluate_grounding([roast()], {"img_1": description()})
    safety = evaluate_safety(load_probes())
    record = roast_metric_record(
        grounding, safety, model="rule_roaster", split="val", config={}
    )
    assert "quality is not" in record["metrics"]["note"]
    assert "No human ratings" in record["metrics"]["note"]


def test_the_metric_record_serialises_without_a_non_finite_value() -> None:
    grounding = evaluate_grounding([roast()], {"img_1": description()})
    safety = evaluate_safety(load_probes())
    record = roast_metric_record(
        grounding, safety, model="rule_roaster", split="val", config={}
    )
    assert json.dumps(record, allow_nan=False)
