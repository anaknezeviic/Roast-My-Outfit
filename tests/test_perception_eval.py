"""Check perception metrics on tiny hand-computed examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rmo import paths
from rmo.eval import metrics as eval_metrics
from rmo.eval.perception_eval import (
    EvaluationResult,
    _read_cache,
    compute_metrics,
    evaluate_predictions,
    evaluate_readouts,
    is_fallback,
    log_evaluation,
    perception_metric_record,
    run_evaluation,
)
from rmo.imaging import ImageInput
from rmo.perception.base import PerceptionModel
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


def label_row(image_id: str, **overrides: object) -> dict[str, object]:
    """Build one label row whose fields all carry the ``na`` vocabulary code."""
    row: dict[str, object] = {
        "image_id": image_id,
        "upper_fabric": "na",
        "lower_fabric": "na",
        "outer_fabric": "na",
        "upper_pattern": "na",
        "lower_pattern": "na",
        "outer_pattern": "na",
        "sleeve_length": "na",
        "lower_length": "na",
        "neckline": "na",
        "has_shape": True,
    }
    row.update(overrides)
    return row


def mixed_supervision_frame() -> pd.DataFrame:
    """Return two rows where only the first carries shape supervision."""
    return pd.DataFrame(
        [
            label_row("supervised", sleeve_length="short", neckline="round"),
            label_row("unsupervised", has_shape=False),
        ]
    )


def mixed_supervision_predictions() -> list[OutfitDescription]:
    """Return one ordinary and one fallback prediction for the mixed frame."""
    return [
        description(
            "supervised",
            [
                Garment(
                    slot=GarmentSlot.upper,
                    category="shirt",
                    sleeve_length=SleeveLength.short,
                    neckline=Neckline.round,
                )
            ],
        ),
        description(
            "unsupervised",
            [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)],
        ),
    ]


def test_compute_metrics_includes_na_in_every_metric() -> None:
    metrics = compute_metrics(
        ["cotton", "cotton", "denim", "na"],
        ["cotton", "denim", "denim", "cotton"],
    )
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(7 / 18)
    assert metrics.n_comparisons == 4
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
                "has_shape": True,
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
                "has_shape": True,
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
    assert result.fields["fabric"].n_comparisons == 6
    assert result.fields["fabric"].accuracy == 1.0
    assert result.fields["sleeve_length"].n_comparisons == 2
    assert result.fields["sleeve_length"].confusion["na", "na"] == 1
    assert result.schema_validity == 0.5
    assert result.sample_size == 2


def test_evaluate_predictions_requires_exact_test_coverage() -> None:
    frame = pd.DataFrame({"image_id": ["one"]})
    with pytest.raises(ValueError, match="labels do not exactly match") as failure:
        evaluate_predictions(frame, [], {"one", "two"})
    assert "has_shape" not in str(failure.value)


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
                "has_shape": True,
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
                "has_shape": True,
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
                "has_shape": True,
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


HEADS = (
    "upper_fabric",
    "lower_fabric",
    "outer_fabric",
    "upper_pattern",
    "lower_pattern",
    "outer_pattern",
    "sleeve_length",
    "lower_length",
    "neckline",
)


def reading(**overrides: str) -> dict[str, str]:
    """Build one raw nine-head decision whose heads all read ``na``."""
    return {head: overrides.get(head, "na") for head in HEADS}


def test_evaluate_readouts_scores_every_head_against_its_label_column() -> None:
    frame = pd.DataFrame([label_row("one", upper_fabric="denim")])
    heads, excluded = evaluate_readouts(
        frame, {"one": reading(upper_fabric="denim", lower_fabric="cotton")}, {"one"}
    )

    assert sorted(heads) == sorted(HEADS)
    assert heads["upper_fabric"].accuracy == 1.0
    assert heads["lower_fabric"].accuracy == 0.0
    assert heads["lower_fabric"].confusion["na", "cotton"] == 1
    assert set(excluded.values()) == {0}


def test_evaluate_readouts_masks_the_shape_heads_on_unsupervised_rows() -> None:
    heads, excluded = evaluate_readouts(
        mixed_supervision_frame(),
        {"supervised": reading(sleeve_length="short"), "unsupervised": reading()},
        {"supervised", "unsupervised"},
    )

    assert heads["sleeve_length"].n_comparisons == 1
    assert heads["sleeve_length"].accuracy == 1.0
    assert excluded["sleeve_length"] == 1
    assert heads["upper_fabric"].n_comparisons == 2
    assert excluded["upper_fabric"] == 0


def test_evaluate_readouts_requires_exact_split_coverage() -> None:
    frame = pd.DataFrame([label_row("one")])
    with pytest.raises(ValueError, match="Readouts do not exactly match"):
        evaluate_readouts(frame, {}, {"one"})


def test_the_metric_record_reports_heads_apart_from_the_projected_fields() -> None:
    result = EvaluationResult(
        fields={"fabric": compute_metrics(["na"], ["na"])},
        schema_validity=1.0,
        valid_generations=1,
        sample_size=1,
        heads={"upper_fabric": compute_metrics(["denim"], ["na"])},
        heads_excluded={"upper_fabric": 0},
    )
    stamp = perception_metric_record(result, model="cnn", config={"a": 1})

    assert sorted(stamp["metrics"]) == ["fields", "heads", "schema_validity"]
    assert stamp["metrics"]["heads"]["upper_fabric"]["accuracy"] == 0.0
    assert stamp["metrics"]["heads"]["upper_fabric"].keys() == (
        stamp["metrics"]["fields"]["fabric"].keys()
    )
    assert "upper_fabric" not in stamp["metrics"]["fields"]


def test_the_metric_record_carries_an_empty_head_block_without_readouts() -> None:
    result = EvaluationResult(
        fields={"fabric": compute_metrics(["na"], ["na"])},
        schema_validity=1.0,
        valid_generations=1,
        sample_size=1,
    )
    assert perception_metric_record(result, model="smol", config={})["metrics"]["heads"] == {}


def test_log_evaluation_reports_the_heads_on_their_own_lines(caplog) -> None:
    result = EvaluationResult(
        fields={},
        schema_validity=1.0,
        valid_generations=1,
        sample_size=1,
        heads={"upper_fabric": compute_metrics(["denim"], ["denim"])},
        heads_excluded={"upper_fabric": 2},
    )
    with caplog.at_level(logging.INFO, logger="rmo.eval.perception_eval"):
        log_evaluation(result)
    output = "\n".join(record.message for record in caplog.records)
    assert "head=upper_fabric accuracy=1.000000 macro_f1=1.000000" in output
    assert "excluded=2" in output


def test_log_evaluation_reports_fields_confusions_and_sample_size(caplog) -> None:
    metrics = compute_metrics(["na"], ["na"])
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


def test_masking_scores_only_the_supervised_shape_rows() -> None:
    result = evaluate_predictions(
        mixed_supervision_frame(),
        mixed_supervision_predictions(),
        {"supervised", "unsupervised"},
    )

    assert result.fields["sleeve_length"].n_comparisons == 1
    assert result.excluded["sleeve_length"] == 1
    assert result.fields["fabric"].n_comparisons == 6
    assert result.excluded["fabric"] == 0
    assert result.sample_size == 2
    assert result.valid_generations == 1
    assert result.schema_validity == 0.5


def test_genuine_na_shape_labels_are_still_scored() -> None:
    frame = pd.DataFrame([label_row("annotated")])
    prediction = description(
        "annotated",
        [Garment(slot=GarmentSlot.other, category="unknown", confidence=0.0)],
    )

    result = evaluate_predictions(frame, [prediction], {"annotated"})

    assert result.fields["sleeve_length"].confusion["na", "na"] == 1
    assert result.excluded["sleeve_length"] == 0
    assert "sleeve_length" not in result.undefined_fields


def test_evaluate_predictions_requires_the_shape_supervision_column() -> None:
    row = label_row("one")
    del row["has_shape"]
    prediction = description("one", [Garment(slot=GarmentSlot.upper, category="shirt")])

    with pytest.raises(ValueError, match="has_shape"):
        evaluate_predictions(pd.DataFrame([row]), [prediction], {"one"})


def test_evaluate_predictions_requires_a_boolean_shape_column() -> None:
    frame = pd.DataFrame([label_row("one", has_shape="False")])
    prediction = description("one", [Garment(slot=GarmentSlot.upper, category="shirt")])

    with pytest.raises(ValueError, match="has_shape must be a boolean column"):
        evaluate_predictions(frame, [prediction], {"one"})


def test_every_masked_row_leaves_the_field_undefined_with_a_reason() -> None:
    frame = pd.DataFrame([label_row("unsupervised", has_shape=False)])
    prediction = description(
        "unsupervised", [Garment(slot=GarmentSlot.upper, category="shirt")]
    )

    result = evaluate_predictions(frame, [prediction], {"unsupervised"})

    assert "sleeve_length" not in result.fields
    assert result.undefined_fields["sleeve_length"] == "no_supervised_rows"
    assert result.excluded["sleeve_length"] == 1
    assert result.fields["fabric"].n_comparisons == 3


def test_log_evaluation_reports_excluded_and_undefined_fields(caplog) -> None:
    result = EvaluationResult(
        fields={"fabric": compute_metrics(["na"], ["na"])},
        schema_validity=1.0,
        valid_generations=1,
        sample_size=1,
        excluded={"fabric": 0, "sleeve_length": 1},
        undefined_fields={"sleeve_length": "no_supervised_rows"},
    )

    with caplog.at_level(logging.INFO, logger="rmo.eval.perception_eval"):
        log_evaluation(result)

    assert ("rmo.eval.perception_eval", "INFO") in [
        (record.name, record.levelname) for record in caplog.records
    ]
    output = "\n".join(record.message for record in caplog.records)
    assert (
        "field=fabric accuracy=1.000000 macro_f1=1.000000 n_comparisons=1 excluded=0" in output
    )
    assert "field=sleeve_length undefined excluded=1 reason=no_supervised_rows" in output


def test_perception_metric_record_reports_per_field_denominators() -> None:
    result = evaluate_predictions(
        mixed_supervision_frame(),
        mixed_supervision_predictions(),
        {"supervised", "unsupervised"},
    )

    record = perception_metric_record(result, model="smol", config={"model_id": "smol"})

    fields = record["metrics"]["fields"]
    assert fields["sleeve_length"]["n"] == 1
    assert fields["sleeve_length"]["excluded"] == 1
    assert fields["fabric"]["n"] == 6
    assert fields["fabric"]["excluded"] == 0
    assert fields["fabric"]["reason"] is None
    assert fields["fabric"]["confusion"] == [{"actual": "na", "count": 6, "predicted": "na"}]
    assert record["metrics"]["schema_validity"] == {"n": 2, "valid": 1, "value": 0.5}
    assert record["stage"] == "perception"
    assert record["n_items"] == 2
    assert json.dumps(record, allow_nan=False)


def test_perception_metric_record_marks_undefined_fields_null_with_a_reason() -> None:
    frame = pd.DataFrame([label_row("unsupervised", has_shape=False)])
    prediction = description(
        "unsupervised", [Garment(slot=GarmentSlot.upper, category="shirt")]
    )
    result = evaluate_predictions(frame, [prediction], {"unsupervised"})

    record = perception_metric_record(result, model="smol", config={"model_id": "smol"})

    fields = record["metrics"]["fields"]
    assert fields["sleeve_length"] == {
        "accuracy": None,
        "confusion": [],
        "excluded": 1,
        "macro_f1": None,
        "n": 0,
        "reason": "no_supervised_rows",
    }
    assert set(fields["sleeve_length"]) == set(fields["fabric"])


def test_perception_metric_record_uses_the_shared_config_hash(monkeypatch) -> None:
    monkeypatch.setattr(eval_metrics, "config_hash", lambda config: "deadbeef")
    frame = pd.DataFrame([label_row("one", sleeve_length="short")])
    prediction = description(
        "one",
        [
            Garment(
                slot=GarmentSlot.upper,
                category="shirt",
                sleeve_length=SleeveLength.short,
            )
        ],
    )
    result = evaluate_predictions(frame, [prediction], {"one"})

    record = perception_metric_record(result, model="smol", config={"model_id": "smol"})

    assert record["config_hash"] == "deadbeef"


def test_perception_metric_record_records_the_split_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path / "data"))
    result = evaluate_predictions(
        mixed_supervision_frame(),
        mixed_supervision_predictions(),
        {"supervised", "unsupervised"},
    )

    unstaged = perception_metric_record(result, model="smol", config={"model_id": "smol"})
    assert unstaged["inputs"] == {}

    split_file = paths.splits_dir() / "test.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_bytes(b"supervised\nunsupervised\n")
    staged = perception_metric_record(result, model="smol", config={"model_id": "smol"})
    assert staged["inputs"] == {"split": eval_metrics.file_sha256(split_file)}


class RecordingModel(PerceptionModel):
    """Report whether the evaluator ever asked for inference."""

    name = "recording"

    def __init__(self) -> None:
        self.called = False

    def predict(self, image: ImageInput) -> OutfitDescription:
        """Note the request and return one trivial description."""
        self.called = True
        return description("any", [Garment(slot=GarmentSlot.upper, category="shirt")])


def test_run_evaluation_rejects_an_unusable_predictions_path_before_inference(
    tmp_path: Path,
) -> None:
    model = RecordingModel()

    with pytest.raises(ValueError, match="must end in .jsonl"):
        run_evaluation(model, predictions_out=tmp_path / "smol_test.json")

    assert not model.called


class ReadoutModel(PerceptionModel):
    """Stand in for an adapter that exposes its raw head decisions."""

    name = "readout"

    def predict(self, image: ImageInput) -> OutfitDescription:
        """Return the single garment the evaluator will score."""
        return description(
            Path(image).stem, [Garment(slot=GarmentSlot.upper, category="shirt")]
        )

    def predict_batch_with_readouts(self, images):
        """Return one description and one raw reading per image."""
        return [
            (
                self.predict(image),
                SimpleNamespace(
                    image_id=Path(image).stem,
                    labels=reading(upper_fabric="denim"),
                ),
            )
            for image in images
        ]


def test_run_evaluation_reports_the_raw_heads_when_the_model_offers_them(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    split_file = tmp_path / "processed" / "splits" / "test.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_bytes(b"supervised\nunsupervised\n")
    table = tmp_path / "outfits.parquet"
    mixed_supervision_frame().to_parquet(table)

    result = run_evaluation(ReadoutModel(), table)

    assert result.heads["upper_fabric"].n_comparisons == 2
    assert result.heads["upper_fabric"].accuracy == 0.0
    assert result.heads["sleeve_length"].n_comparisons == 1
    assert result.heads_excluded["sleeve_length"] == 1
    assert result.fields["fabric"].n_comparisons == 6


def test_run_evaluation_leaves_the_heads_empty_without_readouts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    split_file = tmp_path / "processed" / "splits" / "test.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_bytes(b"supervised\nunsupervised\n")
    table = tmp_path / "outfits.parquet"
    mixed_supervision_frame().to_parquet(table)

    class Plain(ReadoutModel):
        """Offer only the plain batch interface."""

        predict_batch_with_readouts = None

    result = run_evaluation(Plain(), table)

    assert result.heads == {}
    assert result.heads_excluded == {}
    assert result.fields["fabric"].n_comparisons == 6


class CountingModel(PerceptionModel):
    """Record which images were inferred and optionally fail on one of them."""

    name = "counting"

    def __init__(self, fail_on: str | None = None) -> None:
        self.seen: list[str] = []
        self._fail_on = fail_on

    def predict(self, image: ImageInput) -> OutfitDescription:
        """Return one garment, or raise to imitate an interrupted run."""
        image_id = Path(image).stem
        if image_id == self._fail_on:
            raise RuntimeError("inference stopped")
        self.seen.append(image_id)
        return description(image_id, [Garment(slot=GarmentSlot.upper, category="shirt")])


def staged_split(tmp_path: Path) -> Path:
    """Write the two-image test split and label table used by the cache tests."""
    split_file = tmp_path / "processed" / "splits" / "test.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_bytes(b"supervised\nunsupervised\n")
    table = tmp_path / "outfits.parquet"
    mixed_supervision_frame().to_parquet(table)
    return table


def test_an_interrupted_run_keeps_the_predictions_it_finished(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    table = staged_split(tmp_path)
    cache = tmp_path / "cache.tmp"

    with pytest.raises(RuntimeError, match="inference stopped"):
        run_evaluation(CountingModel(fail_on="unsupervised"), table, cache_path=cache)

    survivors = _read_cache(cache)
    assert sorted(survivors) == ["supervised"]


def test_a_resumed_run_only_infers_the_images_the_cache_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    table = staged_split(tmp_path)
    cache = tmp_path / "cache.tmp"
    with pytest.raises(RuntimeError):
        run_evaluation(CountingModel(fail_on="unsupervised"), table, cache_path=cache)

    resumed = CountingModel()
    result = run_evaluation(resumed, table, cache_path=cache)

    assert resumed.seen == ["unsupervised"]
    assert result.sample_size == 2


def test_a_torn_final_cache_line_is_discarded_rather_than_failing_the_run(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    table = staged_split(tmp_path)
    cache = tmp_path / "cache.tmp"
    intact = description("supervised", [Garment(slot=GarmentSlot.upper, category="shirt")])
    cache.write_text(
        f"{intact.model_dump_json()}\n" + intact.model_dump_json()[:40],
        encoding="utf-8",
    )

    resumed = CountingModel()
    run_evaluation(resumed, table, cache_path=cache)

    assert resumed.seen == ["unsupervised"]


def test_publishing_the_predictions_clears_the_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    table = staged_split(tmp_path)
    cache = tmp_path / "cache.tmp"

    run_evaluation(
        CountingModel(),
        table,
        predictions_out=tmp_path / "preds.jsonl",
        cache_path=cache,
    )

    assert (tmp_path / "preds.jsonl").is_file()
    assert not cache.exists()


def test_the_cache_is_untouched_when_no_cache_path_is_given(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    table = staged_split(tmp_path)

    run_evaluation(CountingModel(), table)

    assert list(tmp_path.glob("*.tmp")) == []
