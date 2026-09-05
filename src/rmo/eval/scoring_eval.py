"""Discrimination metrics for compatibility scorers on split-local negatives."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rmo import paths
from rmo.eval.metrics import metric_record, write_metric_record
from rmo.scoring.base import ScoringModel
from rmo.scoring.missingness import area_under_curve
from rmo.scoring.pairs import PairKind, PairRecord, PairSet
from rmo.splits import SPLIT_NAMES

log = logging.getLogger(__name__)

__all__ = [
    "ScoringEvaluation",
    "evaluate_scorer",
    "log_evaluation",
    "scoring_metric_record",
]


@dataclass(frozen=True, slots=True)
class ScoringEvaluation:
    """How well one scorer separates observed outfits from synthesised ones."""

    auc_hard: float | None
    auc_easy: float | None
    auc_pooled: float | None
    accuracy_hard: float | None
    accuracy_easy: float | None
    ranking_accuracy: float | None
    mean_observed: float
    mean_hard: float | None
    mean_easy: float | None
    n_observed: int
    n_hard: int
    n_easy: int
    n_ranked: int

    def as_metrics(self) -> dict[str, Any]:
        """Return the block a metric record carries for this evaluation."""
        return asdict(self)


def _scores(scorer: ScoringModel, records: Sequence[PairRecord]) -> dict[str, float]:
    """Return the overall score of every record, keyed by its description id."""
    return {
        record.description.image_id: float(scorer.score(record.description).overall)
        for record in records
    }


def _paired(
    observed: Mapping[str, float],
    negatives: Sequence[PairRecord],
    values: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the observed and negative score of every recipient that has both."""
    left: list[float] = []
    right: list[float] = []
    for record in negatives:
        if record.recipient_id in observed:
            left.append(observed[record.recipient_id])
            right.append(values[record.description.image_id])
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Return the rank AUC of two score populations, or nothing when either is empty."""
    if positive.size == 0 or negative.size == 0:
        return None
    labels = np.concatenate([np.ones(positive.size), np.zeros(negative.size)]).astype(
        np.int64
    )
    return area_under_curve(labels, np.concatenate([positive, negative]))


def _paired_accuracy(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Return the fraction of recipient pairs the scorer orders correctly."""
    if positive.size == 0:
        return None
    wins = (positive > negative).astype(np.float64)
    ties = (positive == negative).astype(np.float64)
    return float(np.mean(wins + 0.5 * ties))


def evaluate_scorer(scorer: ScoringModel, built: PairSet) -> ScoringEvaluation:
    """Measure how a scorer ranks observed outfits against their negatives."""
    observed_records = built.of_kind(PairKind.observed)
    hard_records = built.of_kind(PairKind.hard)
    easy_records = built.of_kind(PairKind.easy)
    if not observed_records:
        raise ValueError("A scoring evaluation needs at least one observed record.")

    values = _scores(scorer, built.records)
    observed = {
        record.recipient_id: values[record.description.image_id]
        for record in observed_records
    }

    hard_positive, hard_negative = _paired(observed, hard_records, values)
    easy_positive, easy_negative = _paired(observed, easy_records, values)

    pooled_negative = np.concatenate([hard_negative, easy_negative])
    pooled_positive = np.asarray(list(observed.values()), dtype=np.float64)

    hard_by_recipient = {
        record.recipient_id: values[record.description.image_id] for record in hard_records
    }
    easy_by_recipient = {
        record.recipient_id: values[record.description.image_id] for record in easy_records
    }
    ranked = [
        recipient
        for recipient in observed
        if recipient in hard_by_recipient and recipient in easy_by_recipient
    ]
    ordered = sum(
        1
        for recipient in ranked
        if observed[recipient] >= hard_by_recipient[recipient]
        >= easy_by_recipient[recipient]
    )

    return ScoringEvaluation(
        auc_hard=_auc(hard_positive, hard_negative),
        auc_easy=_auc(easy_positive, easy_negative),
        auc_pooled=_auc(pooled_positive, pooled_negative),
        accuracy_hard=_paired_accuracy(hard_positive, hard_negative),
        accuracy_easy=_paired_accuracy(easy_positive, easy_negative),
        ranking_accuracy=float(ordered / len(ranked)) if ranked else None,
        mean_observed=float(np.mean(pooled_positive)),
        mean_hard=float(np.mean(hard_negative)) if hard_negative.size else None,
        mean_easy=float(np.mean(easy_negative)) if easy_negative.size else None,
        n_observed=len(observed_records),
        n_hard=len(hard_records),
        n_easy=len(easy_records),
        n_ranked=len(ranked),
    )


def log_evaluation(result: ScoringEvaluation) -> None:
    """Log one line per measured quantity."""
    log.info(
        "observed=%d hard=%d easy=%d ranked=%d",
        result.n_observed,
        result.n_hard,
        result.n_easy,
        result.n_ranked,
    )
    log.info(
        "auc_hard=%s auc_easy=%s auc_pooled=%s ranking_accuracy=%s",
        result.auc_hard,
        result.auc_easy,
        result.auc_pooled,
        result.ranking_accuracy,
    )


def scoring_metric_record(
    result: ScoringEvaluation,
    *,
    model: str,
    split: str,
    config: Mapping[str, Any],
    seed: int | None = None,
    inputs: Mapping[str, str] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared run stamp around one scoring evaluation."""
    return metric_record(
        stage="scoring",
        model=model,
        split=split,
        n_items=result.n_observed,
        metrics={
            "distribution_fit": result.as_metrics(),
            "note": (
                "Discrimination against synthesised negatives measures distribution fit, "
                "not compatibility. No human validation was collected."
            ),
        },
        config=config,
        seed=seed,
        inputs=inputs,
        baseline=baseline,
    )


def _metrics_filename(model_name: str, split: str) -> str:
    """Return a legal filename for one scorer and split."""
    return f"scoring_{model_name.replace('/', '_')}_{split}.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a registered scorer against split-local negatives."""
    from rmo.data.descriptions import describe_split, load_outfit_table
    from rmo.pipeline import create
    from rmo.scoring.pairs import attrition_report, build_pairs

    parser = argparse.ArgumentParser(description="Evaluate compatibility scoring.")
    parser.add_argument("--scorer", default="rule_scorer_v1")
    parser.add_argument("--split", default="val", choices=SPLIT_NAMES)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    descriptions = list(
        describe_split(args.split, limit=args.limit, table=load_outfit_table(args.table))
    )

    built = build_pairs(descriptions, seed=args.seed)
    scorer = create(args.scorer)
    result = evaluate_scorer(scorer, built)
    log_evaluation(result)

    destination = args.metrics_out or paths.metrics_dir() / _metrics_filename(
        scorer.name, args.split
    )
    write_metric_record(
        scoring_metric_record(
            result,
            model=scorer.name,
            split=args.split,
            config={"pairs": attrition_report(built), "scorer": scorer.name},
            seed=args.seed,
        ),
        destination,
    )
    log.info("wrote scoring metrics to %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
