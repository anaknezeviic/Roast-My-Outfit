"""Orchestrate pair construction, the missingness guard and learned scorer fitting."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rmo import paths
from rmo.eval.metrics import metric_record, write_metric_record
from rmo.schemas import OutfitDescription
from rmo.scoring.features import MAX_MISSINGNESS_AUC, build_spec, feature_contract
from rmo.scoring.learned import LearnedScorer, fit_scorer, save_bundle
from rmo.scoring.missingness import GuardResult, guard_metrics, run_guard
from rmo.scoring.pairs import PairSet, attrition_report, build_pairs, pair_matrix

log = logging.getLogger(__name__)

__all__ = [
    "GuardRefused",
    "SplitPairs",
    "build_split_pairs",
    "coefficient_export",
    "fit_with_guard",
    "guard_record",
    "verify_guard",
]

FIT_SPLIT = "train"
SELECTION_SPLIT = "val"


class GuardRefused(RuntimeError):
    """Raised when learned fitting is attempted without a current guard pass."""


@dataclass(frozen=True, slots=True)
class SplitPairs:
    """Built pairs and their feature matrix for one split."""

    split: str
    built: PairSet
    features: np.ndarray
    labels: np.ndarray
    groups: tuple[str, ...]


def build_split_pairs(
    descriptions: Sequence[OutfitDescription], split: str, spec: Any, *, seed: int
) -> SplitPairs:
    """Build the negatives for one split and turn them into a feature matrix."""
    built = build_pairs(descriptions, seed=seed)
    features, labels, groups = pair_matrix(built.records, spec)
    return SplitPairs(
        split=split, built=built, features=features, labels=labels, groups=tuple(groups)
    )


def verify_guard(fit_pairs: SplitPairs, eval_pairs: SplitPairs, spec: Any, *, seed: int) -> GuardResult:
    """Recompute the guard against the inputs a fit is about to use."""
    if fit_pairs.split != FIT_SPLIT:
        raise ValueError(f"The guard must be fitted on {FIT_SPLIT!r}, got {fit_pairs.split!r}.")
    if eval_pairs.split != SELECTION_SPLIT:
        raise ValueError(
            f"The guard must be evaluated on {SELECTION_SPLIT!r}, got {eval_pairs.split!r}."
        )
    return run_guard(
        fit_pairs.features,
        fit_pairs.labels,
        eval_pairs.features,
        eval_pairs.labels,
        spec,
        seed=seed,
    )


def fit_with_guard(
    fit_pairs: SplitPairs, eval_pairs: SplitPairs, spec: Any, *, seed: int
) -> tuple[Any, GuardResult]:
    """Recompute the guard and fit only when it passes for these exact inputs."""
    result = verify_guard(fit_pairs, eval_pairs, spec, seed=seed)
    if not result.passed:
        raise GuardRefused(
            f"The missingness guard returned {result.status} "
            f"(auc={result.auc}, reason={result.reason}); "
            f"repair pair construction on train and validation and rerun before fitting."
        )
    bundle = fit_scorer(fit_pairs.features, fit_pairs.labels, spec, seed=seed)
    return bundle, result


def guard_record(
    result: GuardResult,
    *,
    model: str,
    config: dict[str, Any],
    seed: int,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return the metric record one guard run writes."""
    return metric_record(
        stage="scoring",
        model=model,
        split=SELECTION_SPLIT,
        n_items=max(result.n_evaluated, 1),
        metrics=guard_metrics(result),
        config=config,
        provenance="gt",
        baseline={"random": 0.5},
        seed=seed,
        inputs=inputs,
    )


def coefficient_export(bundle: Any, spec: Any) -> dict[str, Any]:
    """Return the committed analysis payload a coefficient plot needs."""
    contract = feature_contract(spec)
    return {
        "classes": list(bundle.classes),
        "coefficient_units": "standardised",
        "feature_names": list(bundle.feature_names),
        "feature_version": contract["feature_version"],
        "intercept": bundle.intercept,
        "kind": bundle.kind,
        "n_fit": bundle.n_fit,
        "preprocessing": {"mean": list(bundle.mean), "scale": list(bundle.scale)},
        "seed": bundle.seed,
        "spec_sha256": contract["spec_sha256"],
        "weights": list(bundle.weights),
    }


def _descriptions(split: str, limit: int | None, table: Path | None) -> list[OutfitDescription]:
    """Return ground-truth descriptions for one split."""
    from rmo.data.descriptions import describe_split, load_outfit_table

    return list(describe_split(split, limit=limit, table=load_outfit_table(table)))


def main(argv: Sequence[str] | None = None) -> int:
    """Build pairs, run the guard and optionally fit the learned scorer."""
    parser = argparse.ArgumentParser(description="Train the learned compatibility scorer.")
    parser.add_argument("--build-pairs", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--bundle-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--coefficients-out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if not (args.build_pairs or args.guard or args.fit):
        parser.error("Choose at least one of --build-pairs, --guard or --fit.")

    spec = build_spec()
    pairs = {
        split: build_split_pairs(
            _descriptions(split, args.limit, args.table), split, spec, seed=args.seed
        )
        for split in (FIT_SPLIT, SELECTION_SPLIT)
    }
    for split, built in pairs.items():
        log.info("%s pairs: %s", split, attrition_report(built.built))

    if not (args.guard or args.fit):
        return 0

    config = {
        "features": feature_contract(spec)["spec_sha256"],
        "pairs": {split: attrition_report(built.built) for split, built in pairs.items()},
        "threshold": MAX_MISSINGNESS_AUC,
    }
    result = verify_guard(pairs[FIT_SPLIT], pairs[SELECTION_SPLIT], spec, seed=args.seed)
    destination = args.metrics_out or paths.metrics_dir() / "scoring_missingness_guard.json"
    write_metric_record(
        guard_record(
            result, model="missingness_diagnostic", config=config, seed=args.seed
        ),
        destination,
    )
    log.info("wrote guard evidence to %s", destination)

    if not args.fit:
        return 0 if result.passed else 1
    if not result.passed:
        log.error(
            "refusing to fit: the guard returned %s (auc=%s, reason=%s)",
            result.status,
            result.auc,
            result.reason,
        )
        return 1

    bundle, _ = fit_with_guard(pairs[FIT_SPLIT], pairs[SELECTION_SPLIT], spec, seed=args.seed)
    bundle_path = args.bundle_out or paths.repo_root() / "models" / "scoring" / "logreg.json"
    save_bundle(bundle, bundle_path)
    log.info("wrote scorer bundle to %s", bundle_path)

    coefficients = args.coefficients_out or paths.results_dir() / "metrics" / "scoring_logreg_coefficients.json"
    paths.write_json_atomic(coefficients, coefficient_export(bundle, spec))
    log.info("wrote coefficients to %s", coefficients)

    scorer = LearnedScorer(bundle, spec=spec)
    log.info("fitted %s on %d train rows", scorer.name, bundle.n_fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
