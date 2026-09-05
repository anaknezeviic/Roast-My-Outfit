"""Diagnostic guard against learning the shortcut of which fields are absent."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmo.scoring.features import MAX_MISSINGNESS_AUC, FeatureSpec

log = logging.getLogger(__name__)

__all__ = [
    "ESTIMATOR_SETTINGS",
    "INVERSE_WARNING_AUC",
    "MAX_MISSINGNESS_AUC",
    "PASS",
    "FAIL",
    "INCOMPLETE",
    "GuardResult",
    "area_under_curve",
    "guard_metrics",
    "indicator_columns",
    "indicator_matrix",
    "positive_column",
    "run_guard",
]

INVERSE_WARNING_AUC = 0.4

ESTIMATOR_SETTINGS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
    "penalty": "l2",
    "solver": "lbfgs",
}

PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class GuardResult:
    """What the missingness classifier achieved and whether that clears the gate."""

    status: str
    auc: float | None
    reason: str | None
    threshold: float
    n_fit: int
    n_evaluated: int
    n_indicators: int
    estimator: Mapping[str, Any]
    seed: int
    inverse_warning: bool = False

    @property
    def passed(self) -> bool:
        """Return whether fitting a learned scorer on these inputs is permitted."""
        return self.status == PASS

    def as_metrics(self) -> dict[str, Any]:
        """Return the block a metric record carries for this guard."""
        return {
            "auc": self.auc,
            "estimator": dict(self.estimator),
            "inverse_warning": self.inverse_warning,
            "n_evaluated": self.n_evaluated,
            "n_fit": self.n_fit,
            "n_indicators": self.n_indicators,
            "passed": self.passed,
            "reason": self.reason,
            "seed": self.seed,
            "status": self.status,
            "threshold": self.threshold,
        }


def _incomplete(
    reason: str,
    *,
    threshold: float,
    seed: int,
    n_fit: int,
    n_evaluated: int,
    n_indicators: int,
) -> GuardResult:
    """Return a guard result that reports a null AUC and never passes."""
    log.warning("missingness guard is incomplete: %s", reason)
    return GuardResult(
        status=INCOMPLETE,
        auc=None,
        reason=reason,
        threshold=threshold,
        n_fit=n_fit,
        n_evaluated=n_evaluated,
        n_indicators=n_indicators,
        estimator=dict(ESTIMATOR_SETTINGS),
        seed=seed,
    )


def indicator_columns(spec: FeatureSpec) -> list[int]:
    """Return the positions of the presence and measurement indicator features."""
    return [
        position
        for position, name in enumerate(spec.names)
        if name in spec.indicator_names
    ]


def indicator_matrix(features: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Return only the indicator columns of a feature matrix."""
    if features.ndim != 2:
        raise ValueError(f"A feature matrix must be two dimensional, got {features.ndim}.")
    if features.shape[1] != len(spec.names):
        raise ValueError(
            f"The matrix is {features.shape[1]} features wide, "
            f"but this specification builds {len(spec.names)}."
        )
    columns = indicator_columns(spec)
    if not columns:
        raise ValueError("This specification declares no missingness indicators.")
    selected = features[:, columns]
    if not np.all(np.isfinite(selected)):
        raise ValueError("A missingness indicator is not finite.")
    if not np.all((selected == 0.0) | (selected == 1.0)):
        raise ValueError(
            "A missingness indicator carries a value other than 0 or 1, so an ordinary "
            "attribute column has been selected as a diagnostic predictor."
        )
    return selected


def area_under_curve(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the rank-based AUC, counting ties as half a win."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("An AUC needs both a positive and a negative class.")
    combined = np.concatenate([positives, negatives])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    for value in np.unique(combined):
        tied = combined == value
        ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[: positives.size].sum()
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2)
        / (positives.size * negatives.size)
    )


def positive_column(classes: Sequence[int] | np.ndarray) -> int:
    """Return the ``predict_proba`` column that carries the observed-combination class."""
    ordered = [int(value) for value in np.asarray(classes).tolist()]
    if len(ordered) != 2:
        raise ValueError(f"A binary estimator needs two classes, got {ordered}.")
    if set(ordered) != {0, 1}:
        raise ValueError(f"Labels must be exactly 0 and 1, got {ordered}.")
    return ordered.index(1)


def run_guard(
    fit_features: np.ndarray,
    fit_labels: Sequence[int] | np.ndarray,
    eval_features: np.ndarray,
    eval_labels: Sequence[int] | np.ndarray,
    spec: FeatureSpec,
    *,
    seed: int,
    threshold: float = MAX_MISSINGNESS_AUC,
) -> GuardResult:
    """Fit a classifier on indicators alone and report its validation discrimination."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    fit_y = np.asarray(fit_labels, dtype=np.int64)
    eval_y = np.asarray(eval_labels, dtype=np.int64)
    fit_x = indicator_matrix(fit_features, spec)
    eval_x = indicator_matrix(eval_features, spec)
    shape = {
        "n_fit": int(fit_x.shape[0]),
        "n_evaluated": int(eval_x.shape[0]),
        "n_indicators": int(fit_x.shape[1]),
    }

    if fit_x.shape[0] == 0 or eval_x.shape[0] == 0:
        return _incomplete("no_eligible_rows", threshold=threshold, seed=seed, **shape)
    if len(np.unique(fit_y)) < 2:
        return _incomplete("one_class_fit_rows", threshold=threshold, seed=seed, **shape)
    if len(np.unique(eval_y)) < 2:
        return _incomplete("one_class_evaluation_rows", threshold=threshold, seed=seed, **shape)

    classifier = LogisticRegression(random_state=seed, **ESTIMATOR_SETTINGS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            classifier.fit(fit_x, fit_y)
        except Exception as exc:
            return _incomplete(f"fit_failed: {exc}", threshold=threshold, seed=seed, **shape)
    if any(issubclass(entry.category, ConvergenceWarning) for entry in caught):
        return _incomplete("fit_did_not_converge", threshold=threshold, seed=seed, **shape)

    scores = classifier.predict_proba(eval_x)[:, positive_column(classifier.classes_)]
    if not np.all(np.isfinite(scores)):
        return _incomplete("non_finite_scores", threshold=threshold, seed=seed, **shape)

    auc = area_under_curve(eval_y, scores)
    if not 0.0 <= auc <= 1.0:
        return _incomplete(
            f"auc_outside_unit_interval: {auc}", threshold=threshold, seed=seed, **shape
        )

    inverse = auc < INVERSE_WARNING_AUC
    if inverse:
        log.warning(
            "missingness auc %.4f is below %.2f, which can indicate an inversely predictive "
            "shortcut rather than an absence of leakage",
            auc,
            INVERSE_WARNING_AUC,
        )

    result = GuardResult(
        status=PASS if auc <= threshold else FAIL,
        auc=auc,
        reason=None,
        threshold=threshold,
        estimator=dict(ESTIMATOR_SETTINGS),
        seed=seed,
        inverse_warning=inverse,
        **shape,
    )
    log.info(
        "missingness guard status=%s auc=%.4f threshold=%.2f", result.status, auc, threshold
    )
    return result


def guard_metrics(result: GuardResult) -> dict[str, Any]:
    """Return the metrics mapping a guard record carries, with its chance reference."""
    return {"missingness": result.as_metrics()}
