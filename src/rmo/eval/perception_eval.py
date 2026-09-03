"""Evaluate outfit perception on the held-out dataset split."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rmo import paths
from rmo.perception.base import PerceptionModel
from rmo.schemas import GarmentSlot, OutfitDescription
from rmo.splits import load_split

log = logging.getLogger(__name__)

__all__ = [
    "EvaluationResult",
    "FieldMetrics",
    "compute_metrics",
    "evaluate_predictions",
    "is_fallback",
    "log_evaluation",
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


@dataclass(frozen=True)
class FieldMetrics:
    """Accuracy, macro-F1 and confusion counts for one field."""

    accuracy: float
    macro_f1: float
    sample_size: int
    confusion: dict[tuple[str, str], int]


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and schema validity for one held-out evaluation."""

    fields: dict[str, FieldMetrics]
    schema_validity: float
    valid_generations: int
    sample_size: int


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
        sample_size=len(actual),
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

    pairs: dict[str, tuple[list[str], list[str]]] = {
        field: ([], []) for field in (*_SLOT_FIELDS, *_SHAPE_FIELDS)
    }
    for image_id in sorted(expected_ids):
        row = labels.loc[image_id]
        description = predictions[image_id]
        for field, slot_columns in _SLOT_FIELDS.items():
            actual, predicted = pairs[field]
            for slots, column in slot_columns:
                actual.append(str(row[column]))
                predicted.append(_slot_value(description, slots, field))
        for field, (column, attribute, slots) in _SHAPE_FIELDS.items():
            actual, predicted = pairs[field]
            actual.append(str(row[column]))
            predicted.append(_shape_value(description, attribute, slots))

    field_metrics = {
        field: compute_metrics(actual, predicted)
        for field, (actual, predicted) in pairs.items()
    }
    valid = sum(not is_fallback(predictions[image_id]) for image_id in expected_ids)
    sample_size = len(expected_ids)
    return EvaluationResult(
        fields=field_metrics,
        schema_validity=valid / sample_size if sample_size else 0.0,
        valid_generations=valid,
        sample_size=sample_size,
    )


def log_evaluation(result: EvaluationResult) -> None:
    """Log every metric and confusion count in deterministic order."""
    log.info("perception evaluation sample_size=%d split=test", result.sample_size)
    for field, metrics in result.fields.items():
        log.info(
            "field=%s accuracy=%.6f macro_f1=%.6f comparisons=%d",
            field,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.sample_size,
        )
        for (actual, predicted), count in metrics.confusion.items():
            log.info(
                "field=%s actual=%s predicted=%s count=%d",
                field,
                actual,
                predicted,
                count,
            )
    log.info(
        "schema_validity=%.6f valid=%d",
        result.schema_validity,
        result.valid_generations,
    )


def run_evaluation(
    model: PerceptionModel,
    table_path: Path | None = None,
) -> EvaluationResult:
    """Run a perception model over the test split and log its metrics."""
    test_ids = load_split("test")
    source = table_path or paths.data_root() / "processed" / "outfits.parquet"
    frame = pd.read_parquet(source)
    held_out = frame.loc[frame["image_id"].isin(test_ids)].copy()
    images = [paths.raw_dir() / "images" / f"{image_id}.jpg" for image_id in sorted(test_ids)]
    descriptions = model.predict_batch(images)
    result = evaluate_predictions(held_out, descriptions, test_ids)
    log_evaluation(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured SmolVLM checkpoint on the held-out split."""
    parser = argparse.ArgumentParser(description="Evaluate perception on the test split.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from rmo.perception.vlm import DEFAULT_MODEL_ID, SmolVLMPerception

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    model = SmolVLMPerception(args.model_id or DEFAULT_MODEL_ID, device=args.device)
    run_evaluation(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())