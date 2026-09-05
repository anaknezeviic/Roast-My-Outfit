"""Cover scorer discrimination against split-local negatives."""

from __future__ import annotations

import json

import pytest

from rmo.eval.scoring_eval import (
    ScoringEvaluation,
    evaluate_scorer,
    scoring_metric_record,
)
from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    OutfitDescription,
    OutfitScore,
    Pattern,
    Provenance,
    SubScores,
)
from rmo.scoring.pairs import PairKind, build_pairs

SEED = 20260101


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


def built(count: int = 10):
    return build_pairs([outfit(index) for index in range(1, count + 1)], seed=SEED)


class ScriptedScorer:
    """Assign a fixed score to each record kind."""

    name = "scripted_scorer"

    def __init__(self, observed: float, hard: float, easy: float) -> None:
        self._values = {"observed": observed, "hard": hard, "easy": easy}

    def score(self, description: OutfitDescription) -> OutfitScore:
        kind = description.image_id.partition("#")[2] or "observed"
        return OutfitScore(
            image_id=description.image_id,
            overall=self._values[kind],
            subscores=SubScores(
                color_harmony=50.0,
                formality_consistency=50.0,
                seasonality=50.0,
                proportion=50.0,
            ),
            issues=[],
            provenance=Provenance.predicted,
            source_model=self.name,
        )


def test_a_perfect_ordering_scores_one_everywhere() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    assert result.auc_hard == pytest.approx(1.0)
    assert result.auc_easy == pytest.approx(1.0)
    assert result.ranking_accuracy == pytest.approx(1.0)


def test_a_reversed_ordering_scores_zero() -> None:
    result = evaluate_scorer(ScriptedScorer(10.0, 60.0, 90.0), built())
    assert result.auc_hard == pytest.approx(0.0)
    assert result.auc_easy == pytest.approx(0.0)
    assert result.ranking_accuracy == pytest.approx(0.0)


def test_a_constant_scorer_lands_on_one_half() -> None:
    result = evaluate_scorer(ScriptedScorer(50.0, 50.0, 50.0), built())
    assert result.auc_hard == pytest.approx(0.5)
    assert result.auc_pooled == pytest.approx(0.5)


def test_ties_count_as_half_a_win_in_the_paired_accuracy() -> None:
    result = evaluate_scorer(ScriptedScorer(50.0, 50.0, 50.0), built())
    assert result.accuracy_hard == pytest.approx(0.5)


def test_an_easy_negative_can_be_separated_while_a_hard_one_is_not() -> None:
    result = evaluate_scorer(ScriptedScorer(60.0, 60.0, 10.0), built())
    assert result.auc_hard == pytest.approx(0.5)
    assert result.auc_easy == pytest.approx(1.0)


def test_the_means_follow_the_scripted_values() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    assert result.mean_observed == pytest.approx(90.0)
    assert result.mean_hard == pytest.approx(60.0)
    assert result.mean_easy == pytest.approx(30.0)


def test_the_counts_match_the_built_corpus() -> None:
    corpus = built()
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), corpus)
    assert result.n_observed == len(corpus.of_kind(PairKind.observed))
    assert result.n_hard == len(corpus.of_kind(PairKind.hard))
    assert result.n_easy == len(corpus.of_kind(PairKind.easy))


def test_the_ranked_count_needs_both_negative_kinds() -> None:
    corpus = built()
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), corpus)
    assert result.n_ranked <= min(result.n_hard, result.n_easy)


def test_an_evaluation_needs_an_observed_record() -> None:
    corpus = built()
    empty = corpus.__class__(
        records=tuple(
            record for record in corpus.records if record.kind is not PairKind.observed
        ),
        attrition={},
        swappable=corpus.swappable,
        seed=corpus.seed,
    )
    with pytest.raises(ValueError, match="at least one observed"):
        evaluate_scorer(ScriptedScorer(1.0, 2.0, 3.0), empty)


def test_a_corpus_without_negatives_reports_undefined_discrimination() -> None:
    corpus = built()
    only_observed = corpus.__class__(
        records=corpus.of_kind(PairKind.observed),
        attrition={},
        swappable=corpus.swappable,
        seed=corpus.seed,
    )
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), only_observed)
    assert result.auc_hard is None
    assert result.auc_easy is None
    assert result.ranking_accuracy is None


def test_the_metric_block_is_flat_and_named() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    block = result.as_metrics()
    assert block["n_observed"] == result.n_observed
    assert set(block) == set(ScoringEvaluation.__dataclass_fields__)


def test_the_metric_record_names_the_scoring_stage() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    record = scoring_metric_record(
        result, model="rule_scorer_v1", split="val", config={"scorer": "rule_scorer_v1"}
    )
    assert record["stage"] == "scoring"
    assert record["split"] == "val"
    assert record["model"] == "rule_scorer_v1"
    assert record["n_items"] == result.n_observed


def test_the_metric_record_states_the_distribution_fit_limitation() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    record = scoring_metric_record(
        result, model="rule_scorer_v1", split="val", config={"scorer": "rule_scorer_v1"}
    )
    assert "distribution fit" in record["metrics"]["note"]
    assert "No human validation" in record["metrics"]["note"]


def test_the_metric_record_serialises_without_a_non_finite_value() -> None:
    result = evaluate_scorer(ScriptedScorer(90.0, 60.0, 30.0), built())
    record = scoring_metric_record(
        result, model="rule_scorer_v1", split="val", config={"scorer": "rule_scorer_v1"}
    )
    assert json.dumps(record, allow_nan=False)


def test_the_real_rule_scorer_runs_over_a_built_corpus() -> None:
    from rmo.scoring.rules import RuleScorer

    result = evaluate_scorer(RuleScorer(), built())
    assert result.n_observed == 10
    assert 0.0 <= result.mean_observed <= 100.0
