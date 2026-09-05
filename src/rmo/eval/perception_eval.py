"""Evaluate outfit perception on the held-out dataset split."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from rmo import paths
from rmo.config import load_perception_config
from rmo.eval.metrics import metric_record, split_inputs, write_metric_record
from rmo.eval.predictions import check_prediction_path, write_predictions
from rmo.imaging import image_identity
from rmo.perception.base import PerceptionModel
from rmo.pipeline import create
from rmo.schemas import GarmentSlot, OutfitDescription
from rmo.splits import load_split

log = logging.getLogger(__name__)

__all__ = [
    "EvaluationResult",
    "FieldMetrics",
    "compute_metrics",
    "evaluate_predictions",
    "evaluate_readouts",
    "is_fallback",
    "log_evaluation",
    "perception_metric_record",
    "run_evaluation",
]

_SLOT_FIELDS: dict[str, tuple[tuple[tuple[GarmentSlot, ...], str], ...]] = {
    "fabric": (
        ((GarmentSlot.upper, GarmentSlot.dress, GarmentSlot.romper), "upper_fabric"),
        ((GarmentSlot.lower, GarmentSlot.dress, GarmentSlot.romper), "lower_fabric"),
        ((GarmentSlot.outer,), "outer_fabric"),
    ),
    "pattern": (
        ((GarmentSlot.upper, GarmentSlot.dress, GarmentSlot.romper), "upper_pattern"),
        ((GarmentSlot.lower, GarmentSlot.dress, GarmentSlot.romper), "lower_pattern"),
        ((GarmentSlot.outer,), "outer_pattern"),
    ),
}

_SHAPE_FIELDS: dict[str, tuple[str, str, tuple[GarmentSlot, ...]]] = {
    "sleeve_length": (
        "sleeve_length",
        "sleeve_length",
        (GarmentSlot.dress, GarmentSlot.romper, GarmentSlot.upper, GarmentSlot.outer),
    ),
    "length": (
        "lower_length",
        "length",
        (GarmentSlot.dress, GarmentSlot.romper, GarmentSlot.lower),
    ),
    "neckline": (
        "neckline",
        "neckline",
        (GarmentSlot.dress, GarmentSlot.romper, GarmentSlot.upper, GarmentSlot.outer),
    ),
}

_SHAPE_HEADS = frozenset({"sleeve_length", "lower_length", "neckline"})


@dataclass(frozen=True)
class FieldMetrics:
    """Accuracy, macro-F1 and confusion counts for one field."""

    accuracy: float
    macro_f1: float
    n_comparisons: int
    confusion: dict[tuple[str, str], int]


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and schema validity for one held-out evaluation."""

    fields: dict[str, FieldMetrics]
    schema_validity: float
    valid_generations: int
    sample_size: int
    excluded: dict[str, int] = field(default_factory=dict)
    undefined_fields: dict[str, str] = field(default_factory=dict)
    heads: dict[str, FieldMetrics] = field(default_factory=dict)
    heads_excluded: dict[str, int] = field(default_factory=dict)


def compute_metrics(actual: Sequence[str], predicted: Sequence[str]) -> FieldMetrics:
    """Return hand-computed classification metrics, including ``na`` values."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have the same length.")
    if not actual:
        raise ValueError("At least one classification pair is required.")

    confusion = Counter(zip(actual, predicted))
    labels = sorted(set(actual) | set(predicted))
    correct = sum(confusion[label, label] for label in labels)
    f1_scores: list[float] = []
    for label in labels:
        true_positive = confusion[label, label]
        false_positive = sum(
            count
            for (truth, guess), count in confusion.items()
            if guess == label and truth != label
        )
        false_negative = sum(
            count
            for (truth, guess), count in confusion.items()
            if truth == label and guess != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)

    return FieldMetrics(
        accuracy=correct / len(actual),
        macro_f1=sum(f1_scores) / len(f1_scores),
        n_comparisons=len(actual),
        confusion=dict(sorted(confusion.items())),
    )


def is_fallback(description: OutfitDescription) -> bool:
    """Return whether the description contains the parser's fallback garment."""
    return len(description.garments) == 1 and (
        description.garments[0].slot is GarmentSlot.other
        and description.garments[0].category == "unknown"
        and description.garments[0].confidence == 0.0
    )


def _slot_value(
    description: OutfitDescription,
    slots: tuple[GarmentSlot, ...],
    attribute: str,
) -> str:
    """Return the first applicable slot value, or ``na`` when absent."""
    for slot in slots:
        garments = description.by_slot(slot)
        if garments:
            value = getattr(garments[0], attribute)
            return value.value if value is not None else "na"
    return "na"


def _shape_value(
    description: OutfitDescription,
    attribute: str,
    slots: tuple[GarmentSlot, ...],
) -> str:
    """Return one primary applicable shape value, or ``na`` when absent."""
    for slot in slots:
        garments = description.by_slot(slot)
        if not garments:
            continue
        value = getattr(garments[0], attribute)
        if value is not None:
            return value.value
    return "na"


def _prediction_map(
    descriptions: Sequence[OutfitDescription],
) -> dict[str, OutfitDescription]:
    """Key predictions by image id and reject duplicates."""
    predictions: dict[str, OutfitDescription] = {}
    for description in descriptions:
        if description.image_id in predictions:
            raise ValueError(f"Duplicate prediction for {description.image_id!r}.")
        predictions[description.image_id] = description
    return predictions


def evaluate_predictions(
    frame: pd.DataFrame,
    descriptions: Sequence[OutfitDescription],
    expected_ids: set[str],
) -> EvaluationResult:
    """Evaluate predictions whose ids exactly cover the held-out rows."""
    if frame["image_id"].duplicated().any():
        raise ValueError("Held-out labels contain duplicate image ids.")
    labels = frame.set_index("image_id")
    label_ids = set(labels.index)
    predictions = _prediction_map(descriptions)
    if label_ids != expected_ids:
        raise ValueError("Held-out labels do not exactly match the test split.")
    if set(predictions) != expected_ids:
        raise ValueError("Predictions do not exactly match the test split.")
    if "has_shape" not in frame.columns:
        raise ValueError("Held-out labels need a has_shape column to mask unsupervised heads.")
    if frame["has_shape"].dtype != bool:
        raise ValueError("has_shape must be a boolean column to mask unsupervised heads.")

    pairs: dict[str, tuple[list[str], list[str]]] = {
        field: ([], []) for field in (*_SLOT_FIELDS, *_SHAPE_FIELDS)
    }
    excluded: dict[str, int] = {field: 0 for field in (*_SLOT_FIELDS, *_SHAPE_FIELDS)}
    for image_id in sorted(expected_ids):
        row = labels.loc[image_id]
        description = predictions[image_id]
        for field, slot_columns in _SLOT_FIELDS.items():
            actual, predicted = pairs[field]
            for slots, column in slot_columns:
                actual.append(str(row[column]))
                predicted.append(_slot_value(description, slots, field))
        supervised = bool(row["has_shape"])
        for field, (column, attribute, slots) in _SHAPE_FIELDS.items():
            if not supervised:
                excluded[field] += 1
                continue
            actual, predicted = pairs[field]
            actual.append(str(row[column]))
            predicted.append(_shape_value(description, attribute, slots))

    field_metrics: dict[str, FieldMetrics] = {}
    undefined_fields: dict[str, str] = {}
    for field, (actual, predicted) in pairs.items():
        if actual:
            field_metrics[field] = compute_metrics(actual, predicted)
        else:
            undefined_fields[field] = "no_supervised_rows"
    valid = sum(not is_fallback(predictions[image_id]) for image_id in expected_ids)
    sample_size = len(expected_ids)
    return EvaluationResult(
        fields=field_metrics,
        schema_validity=valid / sample_size if sample_size else 0.0,
        valid_generations=valid,
        sample_size=sample_size,
        excluded=excluded,
        undefined_fields=undefined_fields,
    )


def evaluate_readouts(
    frame: pd.DataFrame,
    readouts: Mapping[str, Mapping[str, str]],
    expected_ids: set[str],
) -> tuple[dict[str, FieldMetrics], dict[str, int]]:
    """Score each raw head against the label column it was trained on."""
    if set(readouts) != expected_ids:
        raise ValueError("Readouts do not exactly match the evaluated split.")
    labels = frame.set_index("image_id")
    heads = sorted({head for reading in readouts.values() for head in reading})
    pairs: dict[str, tuple[list[str], list[str]]] = {head: ([], []) for head in heads}
    excluded: dict[str, int] = {head: 0 for head in heads}
    for image_id in sorted(expected_ids):
        row = labels.loc[image_id]
        reading = readouts[image_id]
        supervised = bool(row["has_shape"])
        for head in heads:
            if head in _SHAPE_HEADS and not supervised:
                excluded[head] += 1
                continue
            actual, predicted = pairs[head]
            actual.append(str(row[head]))
            predicted.append(str(reading[head]))
    measured = {
        head: compute_metrics(actual, predicted)
        for head, (actual, predicted) in pairs.items()
        if actual
    }
    return measured, excluded


def log_evaluation(result: EvaluationResult) -> None:
    """Log every metric and confusion count in deterministic order."""
    log.info("perception evaluation sample_size=%d split=test", result.sample_size)
    for field, metrics in result.fields.items():
        log.info(
            "field=%s accuracy=%.6f macro_f1=%.6f n_comparisons=%d excluded=%d",
            field,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.n_comparisons,
            result.excluded.get(field, 0),
        )
        for (actual, predicted), count in metrics.confusion.items():
            log.info(
                "field=%s actual=%s predicted=%s count=%d",
                field,
                actual,
                predicted,
                count,
            )
    for field, reason in result.undefined_fields.items():
        log.info(
            "field=%s undefined excluded=%d reason=%s",
            field,
            result.excluded.get(field, 0),
            reason,
        )
    for head, metrics in result.heads.items():
        log.info(
            "head=%s accuracy=%.6f macro_f1=%.6f n_comparisons=%d excluded=%d",
            head,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.n_comparisons,
            result.heads_excluded.get(head, 0),
        )
    log.info(
        "schema_validity=%.6f valid=%d",
        result.schema_validity,
        result.valid_generations,
    )


def _metric_block(metrics: FieldMetrics, excluded: int) -> dict[str, Any]:
    """Return one measured per-field block, key-identical to the undefined form."""
    return {
        "accuracy": metrics.accuracy,
        "confusion": [
            {"actual": actual, "count": count, "predicted": predicted}
            for (actual, predicted), count in sorted(metrics.confusion.items())
        ],
        "excluded": excluded,
        "macro_f1": metrics.macro_f1,
        "n": metrics.n_comparisons,
        "reason": None,
    }


def perception_metric_record(
    result: EvaluationResult,
    *,
    model: str,
    config: Mapping[str, Any],
    split: str = "test",
    seed: int | None = None,
) -> dict[str, Any]:
    """Return the shared run stamp for one perception evaluation."""
    fields: dict[str, Any] = {}
    for name, metrics in result.fields.items():
        fields[name] = _metric_block(metrics, result.excluded.get(name, 0))
    for name, reason in result.undefined_fields.items():
        fields[name] = {
            "accuracy": None,
            "confusion": [],
            "excluded": result.excluded.get(name, 0),
            "macro_f1": None,
            "n": 0,
            "reason": reason,
        }

    return metric_record(
        stage="perception",
        model=model,
        split=split,
        n_items=result.sample_size,
        metrics={
            "fields": fields,
            "heads": {
                name: _metric_block(metrics, result.heads_excluded.get(name, 0))
                for name, metrics in result.heads.items()
            },
            "schema_validity": {
                "n": result.sample_size,
                "valid": result.valid_generations,
                "value": result.schema_validity,
            },
        },
        config=config,
        seed=seed,
        inputs=split_inputs(split),
    )


def _run_config(model: PerceptionModel) -> dict[str, Any]:
    """Return the configuration identity stamped into every artefact of one run."""
    return {"model_id": model.name, "perception": load_perception_config()}


def _read_cache(cache_path: Path) -> dict[str, OutfitDescription]:
    """Return the descriptions an interrupted run already appended."""
    if not cache_path.is_file():
        return {}
    lines = cache_path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1]:
        log.warning("discarding %d characters of a torn final cache line", len(lines.pop()))
    cached: dict[str, OutfitDescription] = {}
    for line in lines:
        if not line:
            continue
        record = OutfitDescription.model_validate_json(line)
        cached[record.image_id] = record
    return cached


def _predict_with_cache(
    model: PerceptionModel,
    images: Sequence[Path],
    cache_path: Path,
    *,
    log_every: int = 25,
) -> list[OutfitDescription]:
    """Predict one image at a time, appending each record so a stopped run can resume."""
    cached = _read_cache(cache_path)
    paths.ensure_dir(cache_path.parent)
    total = len(images)
    log.info(
        "resuming from %s with %d of %d images already predicted",
        cache_path,
        len(cached),
        total,
    )
    descriptions: list[OutfitDescription] = []
    started = time.monotonic()
    fresh = 0
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        for position, image in enumerate(images, start=1):
            image_id, _ = image_identity(image)
            description = cached.get(image_id)
            if description is None:
                description = model.predict(image)
                handle.write(f"{description.model_dump_json()}\n")
                handle.flush()
                os.fsync(handle.fileno())
                fresh += 1
            descriptions.append(description)
            if position % log_every == 0 or position == total:
                rate = (time.monotonic() - started) / fresh if fresh else 0.0
                log.info(
                    "predicted %d/%d images at %.2fs each, %.1f minutes left",
                    position,
                    total,
                    rate,
                    rate * (total - position) / 60,
                )
    return descriptions


def run_evaluation(
    model: PerceptionModel,
    table_path: Path | None = None,
    *,
    predictions_out: Path | None = None,
    cache_path: Path | None = None,
) -> EvaluationResult:
    """Run a perception model over the test split and log its metrics."""
    if predictions_out is not None:
        check_prediction_path(predictions_out)
    test_ids = load_split("test")
    source = table_path or paths.data_root() / "processed" / "outfits.parquet"
    frame = pd.read_parquet(source)
    held_out = frame.loc[frame["image_id"].isin(test_ids)].copy()
    images = [paths.raw_dir() / "images" / f"{image_id}.jpg" for image_id in sorted(test_ids)]
    with_readouts = getattr(model, "predict_batch_with_readouts", None)
    readouts: dict[str, Mapping[str, str]] = {}
    if with_readouts is not None:
        paired = with_readouts(images)
        descriptions = [description for description, _ in paired]
        readouts = {readout.image_id: readout.labels for _, readout in paired}
    elif cache_path is not None:
        descriptions = _predict_with_cache(model, images, cache_path)
    else:
        descriptions = model.predict_batch(images)
    if predictions_out is not None:
        write_predictions(
            descriptions,
            predictions_out,
            split="test",
            model=model.name,
            config=_run_config(model),
            expected_ids=test_ids,
        )
        log.info("wrote %d predictions to %s", len(descriptions), predictions_out)
        if cache_path is not None:
            cache_path.unlink(missing_ok=True)
    result = evaluate_predictions(held_out, descriptions, test_ids)
    if readouts:
        heads, heads_excluded = evaluate_readouts(held_out, readouts, test_ids)
        result = replace(result, heads=heads, heads_excluded=heads_excluded)
    log_evaluation(result)
    return result


def _model_slug(model_name: str) -> str:
    """Return a filename-safe form of a model name that may carry path separators."""
    return model_name.replace("/", "_")


def _metrics_filename(model_name: str) -> str:
    """Return a legal filename stem for a model name that may carry path separators."""
    return f"perception_{_model_slug(model_name)}.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Run a registered perception model on the held-out split."""
    from rmo.perception.vlm import REGISTRY_NAME

    parser = argparse.ArgumentParser(description="Evaluate perception on the test split.")
    parser.add_argument("--perception", default=REGISTRY_NAME)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.perception == REGISTRY_NAME:
        from rmo.perception.vlm import DEFAULT_MODEL_ID, SmolVLMPerception

        model: PerceptionModel = SmolVLMPerception(
            args.model_id or DEFAULT_MODEL_ID, device=args.device
        )
    else:
        model = create(args.perception)

    cache_path = args.cache or (
        paths.predictions_dir() / f"{_model_slug(model.name)}_test.cache.tmp"
    )
    if args.fresh:
        cache_path.unlink(missing_ok=True)

    result = run_evaluation(
        model, predictions_out=args.predictions_out, cache_path=cache_path
    )
    destination = args.metrics_out or paths.metrics_dir() / _metrics_filename(model.name)
    write_metric_record(
        perception_metric_record(result, model=model.name, config=_run_config(model)),
        destination,
    )
    log.info("wrote perception metrics to %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
