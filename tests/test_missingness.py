"""Cover the missingness shortcut guard and its gate semantics."""

from __future__ import annotations

import numpy as np
import pytest

from rmo.scoring.features import MAX_MISSINGNESS_AUC, build_spec
from rmo.scoring.missingness import (
    ESTIMATOR_SETTINGS,
    FAIL,
    INCOMPLETE,
    INVERSE_WARNING_AUC,
    PASS,
    GuardResult,
    area_under_curve,
    guard_metrics,
    indicator_columns,
    indicator_matrix,
    positive_column,
    run_guard,
)

SEED = 20260101


def spec():
    return build_spec()


def indicators(rows: int, spec_, *, rng) -> np.ndarray:
    """Return a full-width matrix whose indicator columns are binary."""
    features = rng.random((rows, len(spec_.names)))
    features[:, indicator_columns(spec_)] = rng.integers(
        0, 2, (rows, len(indicator_columns(spec_)))
    ).astype(float)
    return features


def test_the_threshold_comes_from_the_shared_feature_module() -> None:
    from rmo.scoring import missingness

    assert missingness.MAX_MISSINGNESS_AUC is MAX_MISSINGNESS_AUC
    assert MAX_MISSINGNESS_AUC == 0.6


def test_the_indicator_columns_are_a_subset_of_every_feature() -> None:
    built = spec()
    columns = indicator_columns(built)
    assert columns
    assert len(columns) < len(built.names)
    assert all(built.names[position] in built.indicator_names for position in columns)


def test_the_indicator_columns_are_in_feature_order() -> None:
    columns = indicator_columns(spec())
    assert columns == sorted(columns)


def test_the_indicator_matrix_keeps_only_those_columns() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    selected = indicator_matrix(indicators(12, built, rng=rng), built)
    assert selected.shape == (12, len(indicator_columns(built)))


def test_a_matrix_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(ValueError, match="features wide"):
        indicator_matrix(np.zeros((3, 5)), spec())


def test_a_one_dimensional_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="two dimensional"):
        indicator_matrix(np.zeros(5), spec())


def test_an_indicator_that_is_not_binary_is_refused() -> None:
    built = spec()
    features = np.zeros((4, len(built.names)))
    features[0, indicator_columns(built)[0]] = 0.5
    with pytest.raises(ValueError, match="other than 0 or 1"):
        indicator_matrix(features, built)


def test_a_non_finite_indicator_is_refused() -> None:
    built = spec()
    features = np.zeros((4, len(built.names)))
    features[0, indicator_columns(built)[0]] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        indicator_matrix(features, built)


def test_the_positive_column_follows_the_estimator_class_order() -> None:
    assert positive_column([0, 1]) == 1
    assert positive_column([1, 0]) == 0


def test_a_non_binary_class_order_is_refused() -> None:
    with pytest.raises(ValueError, match="two classes"):
        positive_column([0, 1, 2])


def test_labels_other_than_zero_and_one_are_refused() -> None:
    with pytest.raises(ValueError, match="exactly 0 and 1"):
        positive_column([1, 2])


def test_a_perfect_ranking_scores_one() -> None:
    assert area_under_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == (
        pytest.approx(1.0)
    )


def test_a_reversed_ranking_scores_zero() -> None:
    assert area_under_curve(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == (
        pytest.approx(0.0)
    )


def test_a_constant_score_is_exactly_one_half() -> None:
    assert area_under_curve(np.array([0, 1, 0, 1]), np.ones(4)) == pytest.approx(0.5)


def test_an_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both a positive and a negative"):
        area_under_curve(np.ones(4, dtype=np.int64), np.arange(4.0))


def test_uninformative_indicators_pass_the_gate() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        indicators(400, built, rng=rng),
        rng.integers(0, 2, 400),
        indicators(200, built, rng=rng),
        rng.integers(0, 2, 200),
        built,
        seed=SEED,
    )
    assert result.status == PASS
    assert result.passed
    assert result.auc <= MAX_MISSINGNESS_AUC


def test_an_indicator_that_leaks_the_label_fails_the_gate() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    columns = indicator_columns(built)
    fit_y = rng.integers(0, 2, 400)
    eval_y = rng.integers(0, 2, 200)
    fit_x = indicators(400, built, rng=rng)
    eval_x = indicators(200, built, rng=rng)
    fit_x[:, columns[0]] = fit_y
    eval_x[:, columns[0]] = eval_y

    result = run_guard(fit_x, fit_y, eval_x, eval_y, built, seed=SEED)

    assert result.status == FAIL
    assert not result.passed
    assert result.auc > MAX_MISSINGNESS_AUC


def test_a_one_class_fit_is_incomplete_rather_than_a_pass() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        indicators(40, built, rng=rng),
        np.ones(40, dtype=np.int64),
        indicators(20, built, rng=rng),
        rng.integers(0, 2, 20),
        built,
        seed=SEED,
    )
    assert result.status == INCOMPLETE
    assert result.auc is None
    assert result.reason == "one_class_fit_rows"
    assert not result.passed


def test_a_one_class_evaluation_is_incomplete() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        indicators(40, built, rng=rng),
        rng.integers(0, 2, 40),
        indicators(20, built, rng=rng),
        np.zeros(20, dtype=np.int64),
        built,
        seed=SEED,
    )
    assert result.status == INCOMPLETE
    assert result.reason == "one_class_evaluation_rows"


def test_no_rows_at_all_is_incomplete() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        np.zeros((0, len(built.names))),
        np.zeros(0, dtype=np.int64),
        indicators(20, built, rng=rng),
        rng.integers(0, 2, 20),
        built,
        seed=SEED,
    )
    assert result.status == INCOMPLETE
    assert result.reason == "no_eligible_rows"


def test_an_incomplete_result_is_never_a_pass() -> None:
    result = GuardResult(
        status=INCOMPLETE,
        auc=None,
        reason="one_class_fit_rows",
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=0,
        n_evaluated=0,
        n_indicators=0,
        estimator=ESTIMATOR_SETTINGS,
        seed=SEED,
    )
    assert not result.passed


def test_the_gate_fails_exactly_above_the_threshold() -> None:
    def at(auc: float) -> GuardResult:
        return GuardResult(
            status=PASS if auc <= MAX_MISSINGNESS_AUC else FAIL,
            auc=auc,
            reason=None,
            threshold=MAX_MISSINGNESS_AUC,
            n_fit=1,
            n_evaluated=1,
            n_indicators=1,
            estimator=ESTIMATOR_SETTINGS,
            seed=SEED,
        )

    assert at(MAX_MISSINGNESS_AUC).passed
    assert not at(MAX_MISSINGNESS_AUC + 0.001).passed


def test_an_inverse_shortcut_is_warned_about_rather_than_called_clean() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    columns = indicator_columns(built)
    fit_y = rng.integers(0, 2, 400)
    eval_y = rng.integers(0, 2, 200)
    fit_x = indicators(400, built, rng=rng)
    eval_x = indicators(200, built, rng=rng)
    fit_x[:, columns[0]] = fit_y
    eval_x[:, columns[0]] = 1 - eval_y

    result = run_guard(fit_x, fit_y, eval_x, eval_y, built, seed=SEED)

    assert result.auc < INVERSE_WARNING_AUC
    assert result.inverse_warning
    assert result.status == PASS


def test_the_recorded_estimator_settings_are_explicit() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        indicators(120, built, rng=rng),
        rng.integers(0, 2, 120),
        indicators(60, built, rng=rng),
        rng.integers(0, 2, 60),
        built,
        seed=SEED,
    )
    assert set(result.estimator) == {"C", "max_iter", "penalty", "solver"}
    assert result.seed == SEED


def test_the_guard_reports_the_shapes_it_measured() -> None:
    built = spec()
    rng = np.random.default_rng(SEED)
    result = run_guard(
        indicators(120, built, rng=rng),
        rng.integers(0, 2, 120),
        indicators(60, built, rng=rng),
        rng.integers(0, 2, 60),
        built,
        seed=SEED,
    )
    assert result.n_fit == 120
    assert result.n_evaluated == 60
    assert result.n_indicators == len(indicator_columns(built))


def test_the_metric_block_names_every_reported_quantity() -> None:
    result = GuardResult(
        status=PASS,
        auc=0.55,
        reason=None,
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=10,
        n_evaluated=5,
        n_indicators=3,
        estimator=ESTIMATOR_SETTINGS,
        seed=SEED,
    )
    block = result.as_metrics()
    assert block["status"] == PASS
    assert block["auc"] == 0.55
    assert block["passed"] is True
    assert set(block) == {
        "auc",
        "estimator",
        "inverse_warning",
        "n_evaluated",
        "n_fit",
        "n_indicators",
        "passed",
        "reason",
        "seed",
        "status",
        "threshold",
    }


def test_the_guard_metrics_wrap_the_block_under_its_own_name() -> None:
    result = GuardResult(
        status=PASS,
        auc=0.5,
        reason=None,
        threshold=MAX_MISSINGNESS_AUC,
        n_fit=1,
        n_evaluated=1,
        n_indicators=1,
        estimator=ESTIMATOR_SETTINGS,
        seed=SEED,
    )
    assert set(guard_metrics(result)) == {"missingness"}
