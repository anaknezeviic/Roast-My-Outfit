"""Export perception predictions for every frozen split."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rmo import paths
from rmo.config import load_perception_config
from rmo.eval.predictions import check_prediction_path, write_predictions
from rmo.perception.base import PerceptionModel
from rmo.pipeline import create
from rmo.splits import SPLIT_NAMES, load_split

log = logging.getLogger(__name__)

__all__ = ["export_split", "export_splits", "prediction_path", "run_config"]


def run_config(model: PerceptionModel) -> dict[str, Any]:
    """Return the configuration identity stamped into every exported file."""
    return {"model_id": model.name, "perception": load_perception_config()}


def prediction_path(model_name: str, split: str, directory: Path | None = None) -> Path:
    """Return the export destination for one model and one split."""
    stem = model_name.replace("/", "_")
    base = paths.predictions_dir() if directory is None else Path(directory)
    return base / f"{stem}_{split}.jsonl"


def export_split(
    model: PerceptionModel,
    split: str,
    destination: Path,
    *,
    config: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> Path:
    """Run ``model`` over one split and write its predictions with a manifest."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {', '.join(SPLIT_NAMES)}.")
    check_prediction_path(destination)

    members = sorted(load_split(split))
    image_ids = members if limit is None else members[:limit]
    if not image_ids:
        raise ValueError(f"Split {split!r} holds no image ids to export.")

    images = [paths.raw_dir() / "images" / f"{image_id}.jpg" for image_id in image_ids]
    absent = next((photo for photo in images if not photo.is_file()), None)
    if absent is not None:
        raise FileNotFoundError(f"No photograph at {absent}; stage the images before exporting.")

    log.info("predicting %d %s images with %s", len(images), split, model.name)
    descriptions = model.predict_batch(images)
    return write_predictions(
        descriptions,
        destination,
        split=split,
        model=model.name,
        config=dict(run_config(model) if config is None else config),
        expected_ids=set(image_ids) if limit is None else None,
    )


def export_splits(
    model: PerceptionModel,
    splits: Sequence[str] = SPLIT_NAMES,
    *,
    directory: Path | None = None,
    limit: int | None = None,
) -> dict[str, Path]:
    """Export one prediction file per requested split and return their paths."""
    config = run_config(model)
    written: dict[str, Path] = {}
    for split in splits:
        destination = prediction_path(model.name, split, directory)
        written[split] = export_split(
            model, split, destination, config=config, limit=limit
        )
        log.info("wrote %s predictions to %s", split, written[split])
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Export predictions for the requested splits."""
    parser = argparse.ArgumentParser(description="Export perception predictions.")
    parser.add_argument("--perception", default="cnn_multihead_v1")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES), choices=SPLIT_NAMES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    model = create(args.perception)
    written = export_splits(
        model, args.splits, directory=args.out_dir, limit=args.limit
    )
    for split, destination in written.items():
        log.info("%s -> %s", split, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
