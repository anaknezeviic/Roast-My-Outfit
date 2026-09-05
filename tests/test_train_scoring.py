"""Cover guard-gated learned scorer fitting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    OutfitDescription,
    Pattern,
    Provenance,
)
from rmo.scoring import train_scoring
from rmo.scoring.features import MAX_MISSINGNESS_AUC, build_spec
from rmo.scoring.missingness import FAIL, INCOMPLETE, PASS, GuardResult, indicator_columns
from rmo.scoring.train_scoring import (
    GuardRefused,
    build_split_pairs,
    coefficient_export,
    fit_with_guard,
    guard_record,
    verify_guard,
)

SEED = 20260101


def spec():
    return build_spec()


def outfit(index: int) -> OutfitDescription:
    image_id = f"WOMEN-Blouses_Shirts-id_{index:08d}-01_1_front"
    palette = list(ColorName)
    return OutfitDescription(
        image_id=image_id,
        image_path=f"data/raw/images/{image_id}.jpg",
        source_model="dataset_labels",
        provenance=Provenance.gt,
        garments=[
            Garment(
                slot=slot,
                category=slot.value,
                color=palette[(index + offset) % len(palette)],
                pattern=Pattern.pure_color,
                fabric=Fabric.cotton,
                area_fraction=0.3,
            )
            for offset, slot in enumerate(
                (GarmentSlot.upper, GarmentSlot.lower, GarmentSlot.footwear)
            )
        ],
    )


def pairs(split: str, start: int, count: int):
    return build_split_pairs(
        [outfit(index) for index in range(start, start + count)], split, spec(), seed=SEED
    )


def test_building_returns_a_matrix_matching_the_records() -> None:
    built = pairs("train", 1, 12)
    assert built.features.shape[0] == len(built.built.records)
    assert built.labels.shape[0] == built.features.shape[0]
    assert len(built.groups) == built.features.shape[0]


def test_the_guard_must_be_fitted_on_train() -> None:
    with pytest.raises(ValueError, match="must be fitted on 'train'"):
        verify_guard(pairs("val", 1, 8), pairs("val", 20, 6), spec(), seed=SEED)


def test_the_guard_must_be_evaluated_on_validation() -> None:
    with pytest.raises(ValueError, match="must be evaluated on 'val'"):
        verify_guard(pairs("train", 1, 8), pairs("train", 20, 6), spec(), seed=SEED)


def test_a_guard_run_reports_one_of_the_three_states() -> None:
    result = verify_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    assert result.status in {PASS, FAIL, INCOMPLETE}
    assert result.threshold == MAX_MISSINGNESS_AUC


def test_fitting_is_refused_when_the_guard_does_not_pass(monkeypatch) -> None:
    refused = GuardResult(
        status=FAIL,
        auc=0.91,
        reason=None,
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=10,
        n_evaluated=5,
        n_indicators=3,
        estimator={},
        seed=SEED,
    )
    monkeypatch.setattr(train_scoring, "verify_guard", lambda *a, **k: refused)
    with pytest.raises(GuardRefused, match="FAIL"):
        fit_with_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)


def test_fitting_is_refused_when_the_guard_is_incomplete(monkeypatch) -> None:
    blocked = GuardResult(
        status=INCOMPLETE,
        auc=None,
        reason="one_class_fit_rows",
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=0,
        n_evaluated=0,
        n_indicators=0,
        estimator={},
        seed=SEED,
    )
    monkeypatch.setattr(train_scoring, "verify_guard", lambda *a, **k: blocked)
    with pytest.raises(GuardRefused, match="one_class_fit_rows"):
        fit_with_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)


def test_fitting_recomputes_the_guard_rather_than_trusting_a_stored_value(monkeypatch) -> None:
    calls: list[int] = []

    def recording(*args, **kwargs):
        calls.append(1)
        return GuardResult(
            status=PASS,
            auc=0.5,
            reason=None,
            threshold=MAX_MISSINGNESS_AUC,
            n_fit=10,
            n_evaluated=5,
            n_indicators=3,
            estimator={},
            seed=SEED,
        )

    monkeypatch.setattr(train_scoring, "verify_guard", recording)
    fit_with_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    assert calls == [1]


def test_a_passing_guard_yields_a_fitted_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        train_scoring,
        "verify_guard",
        lambda *a, **k: GuardResult(
            status=PASS,
            auc=0.5,
            reason=None,
            threshold=MAX_MISSINGNESS_AUC,
            n_fit=10,
            n_evaluated=5,
            n_indicators=3,
            estimator={},
            seed=SEED,
        ),
    )
    bundle, result = fit_with_guard(
        pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED
    )
    assert result.passed
    assert len(bundle.weights) == len(spec().names)
    assert bundle.classes == (0, 1)


def test_the_guard_record_names_the_scoring_stage_and_validation_split() -> None:
    result = verify_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    record = guard_record(result, model="missingness_diagnostic", config={}, seed=SEED)
    assert record["stage"] == "scoring"
    assert record["split"] == "val"
    assert record["model"] == "missingness_diagnostic"


def test_the_guard_record_carries_the_analytic_chance_reference() -> None:
    result = verify_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    record = guard_record(result, model="missingness_diagnostic", config={}, seed=SEED)
    assert record["baseline"] == {"random": 0.5}


def test_the_guard_record_serialises_without_a_non_finite_value() -> None:
    result = verify_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    record = guard_record(result, model="missingness_diagnostic", config={}, seed=SEED)
    assert json.dumps(record, allow_nan=False)


def test_an_incomplete_guard_record_reports_a_null_auc_with_a_reason() -> None:
    blocked = GuardResult(
        status=INCOMPLETE,
        auc=None,
        reason="one_class_fit_rows",
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=0,
        n_evaluated=0,
        n_indicators=0,
        estimator={},
        seed=SEED,
    )
    record = guard_record(blocked, model="missingness_diagnostic", config={}, seed=SEED)
    block = record["metrics"]["missingness"]
    assert block["auc"] is None
    assert block["reason"] == "one_class_fit_rows"
    assert block["passed"] is False


def test_the_coefficient_export_labels_its_units_and_order(monkeypatch) -> None:
    monkeypatch.setattr(
        train_scoring,
        "verify_guard",
        lambda *a, **k: GuardResult(
            status=PASS,
            auc=0.5,
            reason=None,
            threshold=MAX_MISSINGNESS_AUC,
            n_fit=10,
            n_evaluated=5,
            n_indicators=3,
            estimator={},
            seed=SEED,
        ),
    )
    bundle, _ = fit_with_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    export = coefficient_export(bundle, spec())

    assert export["coefficient_units"] == "standardised"
    assert export["feature_names"] == list(spec().names)
    assert len(export["weights"]) == len(spec().names)
    assert export["classes"] == [0, 1]
    assert set(export["preprocessing"]) == {"mean", "scale"}


def test_the_coefficient_export_is_json_serialisable(monkeypatch) -> None:
    monkeypatch.setattr(
        train_scoring,
        "verify_guard",
        lambda *a, **k: GuardResult(
            status=PASS,
            auc=0.5,
            reason=None,
            threshold=MAX_MISSINGNESS_AUC,
            n_fit=10,
            n_evaluated=5,
            n_indicators=3,
            estimator={},
            seed=SEED,
        ),
    )
    bundle, _ = fit_with_guard(pairs("train", 1, 12), pairs("val", 40, 8), spec(), seed=SEED)
    assert json.dumps(coefficient_export(bundle, spec()), allow_nan=False)


def test_the_pair_features_carry_only_binary_indicators() -> None:
    built = pairs("train", 1, 12)
    columns = indicator_columns(spec())
    selected = built.features[:, columns]
    assert np.all((selected == 0.0) | (selected == 1.0))
