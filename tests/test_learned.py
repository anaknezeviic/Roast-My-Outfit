"""Cover fitted scorer bundles and the learned overall."""

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
from rmo.scoring.features import build_spec, describe_to_features
from rmo.scoring.learned import (
    BUNDLE_VERSION,
    BundleError,
    LearnedScorer,
    fit_scorer,
    load_bundle,
    save_bundle,
)
from rmo.scoring.pairs import build_pairs, pair_matrix

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


def fitted(kind: str = "logreg"):
    built = build_pairs([outfit(index) for index in range(1, 13)], seed=SEED)
    features, labels, _ = pair_matrix(built.records, spec())
    return fit_scorer(features, labels, spec(), kind=kind, seed=SEED), features, labels


def test_a_bundle_records_the_feature_order_it_was_fitted_under() -> None:
    bundle, _, _ = fitted()
    assert bundle.feature_names == tuple(spec().names)
    assert len(bundle.weights) == len(spec().names)


def test_a_bundle_records_the_rows_it_saw() -> None:
    bundle, features, _ = fitted()
    assert bundle.n_fit == features.shape[0]
    assert bundle.seed == SEED


def test_probabilities_stay_inside_the_unit_interval() -> None:
    bundle, features, _ = fitted()
    probabilities = bundle.probability(features)
    assert probabilities.shape == (features.shape[0],)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_one_row_yields_one_probability() -> None:
    bundle, features, _ = fitted()
    assert bundle.probability(features[0]).shape == (1,)


def test_a_row_of_the_wrong_width_is_refused() -> None:
    bundle, _, _ = fitted()
    with pytest.raises(BundleError, match="expects"):
        bundle.probability(np.zeros((2, 5)))


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown scorer"):
        fit_scorer(np.zeros((4, len(spec().names))), [0, 1, 0, 1], spec(), kind="forest", seed=SEED)


def test_fitting_needs_both_classes() -> None:
    built = spec()
    with pytest.raises(ValueError, match="both an observed"):
        fit_scorer(np.zeros((4, len(built.names))), [1, 1, 1, 1], built, seed=SEED)


def test_fitting_refuses_a_mismatched_label_count() -> None:
    built = spec()
    with pytest.raises(ValueError, match="do not match"):
        fit_scorer(np.zeros((4, len(built.names))), [0, 1], built, seed=SEED)


def test_fitting_refuses_a_matrix_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="features wide"):
        fit_scorer(np.zeros((4, 3)), [0, 1, 0, 1], spec(), seed=SEED)


def test_one_seed_fits_one_bundle() -> None:
    first, _, _ = fitted()
    second, _, _ = fitted()
    assert first.weights == second.weights
    assert first.intercept == second.intercept


def test_a_bundle_round_trips_through_a_file(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    restored = load_bundle(destination, spec())
    assert restored.weights == bundle.weights
    assert restored.intercept == bundle.intercept
    assert restored.feature_names == bundle.feature_names


def test_a_round_tripped_bundle_scores_identically(tmp_path) -> None:
    bundle, features, _ = fitted()
    restored = load_bundle(save_bundle(bundle, tmp_path / "logreg.json"), spec())
    assert np.allclose(restored.probability(features), bundle.probability(features))


def test_a_saved_bundle_is_sorted_json(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    assert payload["bundle_version"] == BUNDLE_VERSION


def test_a_bundle_missing_a_key_is_refused(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    del payload["weights"]
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="missing weights"):
        load_bundle(destination, spec())


def test_a_bundle_from_a_newer_writer_is_refused(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["bundle_version"] = BUNDLE_VERSION + 1
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="declares version"):
        load_bundle(destination, spec())


def test_a_bundle_whose_contract_no_longer_holds_is_refused(tmp_path) -> None:
    from rmo.scoring.features import ContractMismatch, build_spec as _build_spec

    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["contract"]["feature_version"] = "99.0.0"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractMismatch, match="does not hash to its own"):
        load_bundle(destination, _build_spec())


def test_a_bundle_with_a_reordered_feature_list_is_refused(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["feature_names"] = list(reversed(payload["feature_names"]))
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="another feature order"):
        load_bundle(destination, spec())


def test_a_bundle_with_inconsistent_widths_is_refused(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["mean"] = payload["mean"][:-1]
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="inconsistent vector widths"):
        load_bundle(destination, spec())


def test_a_bundle_records_the_estimator_class_order() -> None:
    bundle, _, _ = fitted()
    assert bundle.classes == (0, 1)


def test_the_class_order_decides_which_probability_is_reported() -> None:
    from dataclasses import replace

    bundle, features, _ = fitted()
    flipped = replace(bundle, classes=(1, 0))
    assert flipped.probability(features) == pytest.approx(
        1.0 - bundle.probability(features)
    )


def test_a_bundle_without_a_fitted_scale_is_refused(tmp_path) -> None:
    bundle, _, _ = fitted()
    destination = save_bundle(bundle, tmp_path / "logreg.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["scale"] = [0.0] * len(payload["scale"])
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="non-positive scale"):
        load_bundle(destination, spec())


def test_a_non_finite_feature_row_is_refused() -> None:
    bundle, features, _ = fitted()
    broken = features[:1].copy()
    broken[0, 0] = float("nan")
    with pytest.raises(BundleError, match="non-finite"):
        bundle.probability(broken)


def test_a_bundle_that_is_not_json_is_refused(tmp_path) -> None:
    destination = tmp_path / "logreg.json"
    destination.write_text("not json", encoding="utf-8")
    with pytest.raises(BundleError, match="Could not read"):
        load_bundle(destination, spec())


def test_the_learned_scorer_reports_a_percentage_overall() -> None:
    bundle, _, _ = fitted()
    scorer = LearnedScorer(bundle, spec=spec())
    score = scorer.score(outfit(3))
    assert 0.0 <= score.overall <= 100.0


def test_the_learned_overall_is_the_probability_times_one_hundred() -> None:
    bundle, _, _ = fitted()
    scorer = LearnedScorer(bundle, spec=spec())
    description = outfit(3)
    expected = 100.0 * float(
        bundle.probability(describe_to_features(description, spec()))[0]
    )
    assert scorer.score(description).overall == pytest.approx(expected, abs=1e-4)


def test_the_learned_scorer_keeps_the_rule_subscores_and_issues() -> None:
    from rmo.scoring.rules import RuleScorer

    bundle, _, _ = fitted()
    description = outfit(3)
    rules = RuleScorer().score(description)
    learned = LearnedScorer(bundle, spec=spec()).score(description)
    assert learned.subscores == rules.subscores
    assert learned.issues == rules.issues


def test_the_learned_scorer_names_itself_as_the_source() -> None:
    bundle, _, _ = fitted()
    assert LearnedScorer(bundle, spec=spec()).score(outfit(3)).source_model == (
        LearnedScorer.name
    )


def test_the_learned_scorer_does_not_mutate_the_rule_score() -> None:
    from rmo.scoring.rules import RuleScorer

    bundle, _, _ = fitted()
    description = outfit(3)
    before = RuleScorer().score(description).model_dump_json()
    LearnedScorer(bundle, spec=spec()).score(description)
    assert RuleScorer().score(description).model_dump_json() == before
