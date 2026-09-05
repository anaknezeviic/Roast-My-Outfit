"""Head training for the frozen-backbone attribute model."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from rmo import paths
from rmo.config import load_perception_config
from rmo.eval.metrics import metric_record, write_metric_record
from rmo.perception.cnn import (
    CLASS_MAPS,
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    MultiHeadCNN,
    OutfitAttributeDataset,
    build_backbone,
    build_transform,
    class_weights,
    masked_cross_entropy,
    save_checkpoint,
    split_dataset,
    untrained_heads,
)
from rmo.splits import load_split

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TRAINING",
    "FeatureBank",
    "TrainingRun",
    "extract_features",
    "fit_heads",
    "head_scores",
    "resume_path",
    "train",
]

DEFAULT_TRAINING: dict[str, Any] = {
    "backbone": "resnet50",
    "batch_size": 256,
    "epochs": 30,
    "extract_batch_size": 32,
    "learning_rate": 0.01,
    "num_workers": 4,
    "seed": 20260101,
    "weight_decay": 0.0001,
}

_RESUME_SUFFIX = ".resume.pt"


@dataclass(frozen=True, slots=True)
class FeatureBank:
    """Backbone activations and targets for one split, extracted once."""

    image_ids: tuple[str, ...]
    features: Tensor
    flipped: Tensor | None
    targets: Tensor
    supervised: Tensor

    def __len__(self) -> int:
        """Return how many photographs the bank holds."""
        return len(self.image_ids)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Everything one completed training produces."""

    checkpoint: Path
    best_epoch: int
    best_score: float
    history: tuple[dict[str, Any], ...]
    scores: dict[str, dict[str, float | int]]
    untrained: tuple[str, ...]


def resume_path(checkpoint: Path) -> Path:
    """Return the sidecar that lets an interrupted run continue."""
    return checkpoint.with_name(f"{checkpoint.name}{_RESUME_SUFFIX}")


def _training_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the training block with every absent key defaulted."""
    return {**DEFAULT_TRAINING, **dict(config.get("training", {}))}


def _resolve_device(requested: str | None) -> torch.device:
    """Return the requested device, else CUDA when it is available."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: OutfitAttributeDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    with_flip: bool = False,
) -> FeatureBank:
    """Run the frozen backbone once over ``dataset`` and keep its activations."""
    backbone = backbone.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    upright: list[Tensor] = []
    mirrored: list[Tensor] = []
    for position, batch in enumerate(loader, start=1):
        pixels = batch["pixels"].to(device, non_blocking=True)
        upright.append(backbone(pixels).detach().float().cpu())
        if with_flip:
            mirrored.append(backbone(torch.flip(pixels, dims=(-1,))).detach().float().cpu())
        if position % 20 == 0:
            log.info("extracted %d of %d batches", position, len(loader))

    return FeatureBank(
        image_ids=tuple(dataset.image_ids),
        features=torch.cat(upright),
        flipped=torch.cat(mirrored) if with_flip else None,
        targets=torch.from_numpy(dataset.targets.copy()),
        supervised=torch.from_numpy(dataset.supervised.copy()),
    )


def _macro_f1(actual: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    """Return the unweighted mean F1 over the classes either array uses."""
    scores: list[float] = []
    for index in range(class_count):
        is_actual = actual == index
        is_predicted = predicted == index
        if not is_actual.any() and not is_predicted.any():
            continue
        true_positive = float(np.count_nonzero(is_actual & is_predicted))
        denominator = true_positive + 0.5 * float(
            np.count_nonzero(is_actual ^ is_predicted)
        )
        scores.append(true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def head_scores(
    predicted: Tensor, targets: Tensor, supervised: Tensor
) -> dict[str, dict[str, float | int]]:
    """Return accuracy, macro F1 and the comparison count of every head."""
    scores: dict[str, dict[str, float | int]] = {}
    for index, head in enumerate(HEAD_NAMES):
        rows = supervised[:, index].numpy()
        actual = targets[:, index].numpy()[rows]
        guess = predicted[:, index].numpy()[rows]
        if actual.size == 0:
            scores[head] = {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}
            continue
        scores[head] = {
            "accuracy": float(np.mean(actual == guess)),
            "macro_f1": _macro_f1(actual, guess, HEAD_CLASS_COUNTS[head]),
            "n": int(actual.size),
        }
    return scores


def _selection_score(scores: Mapping[str, Mapping[str, float | int]]) -> float:
    """Return the mean macro F1 over the heads that had supervised rows."""
    measured = [float(block["macro_f1"]) for block in scores.values() if block["n"]]
    return float(np.mean(measured)) if measured else 0.0


def _build_heads(feature_dim: int) -> nn.ModuleDict:
    """Return one linear head per attribute."""
    return nn.ModuleDict(
        {head: nn.Linear(feature_dim, HEAD_CLASS_COUNTS[head]) for head in HEAD_NAMES}
    )


def _logits(heads: nn.ModuleDict, features: Tensor) -> dict[str, Tensor]:
    """Return one logit tensor per head for a batch of backbone activations."""
    return {head: heads[head](features) for head in HEAD_NAMES}


@torch.no_grad()
def _predict(
    heads: nn.ModuleDict, bank: FeatureBank, *, device: torch.device, batch_size: int
) -> Tensor:
    """Return the argmax class index of every head for every row of ``bank``."""
    heads.eval()
    chunks: list[Tensor] = []
    for start in range(0, len(bank), batch_size):
        features = bank.features[start : start + batch_size].to(device)
        logits = _logits(heads, features)
        chunks.append(
            torch.stack([logits[head].argmax(dim=1) for head in HEAD_NAMES], dim=1).cpu()
        )
    return torch.cat(chunks)


def fit_heads(
    train_bank: FeatureBank,
    val_bank: FeatureBank,
    weights: Mapping[str, Sequence[float]],
    *,
    device: torch.device,
    settings: Mapping[str, Any],
    flip_probability: float,
    checkpoint: Path | None = None,
) -> tuple[nn.ModuleDict, int, float, list[dict[str, Any]]]:
    """Train the heads on cached activations, selecting the best epoch on validation."""
    feature_dim = train_bank.features.shape[1]
    torch.manual_seed(int(settings["seed"]))
    heads = _build_heads(feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(settings["seed"]))

    first_epoch = 1
    best_score = -1.0
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in heads.state_dict().items()}
    history: list[dict[str, Any]] = []

    sidecar = resume_path(checkpoint) if checkpoint is not None else None
    if sidecar is not None and sidecar.is_file():
        saved = torch.load(sidecar, map_location="cpu", weights_only=True)
        heads.load_state_dict(saved["heads"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        first_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state"]
        history = list(saved["history"])
        log.info("resuming from %s at epoch %d", sidecar, first_epoch)

    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    for epoch in range(first_epoch, epochs + 1):
        heads.train()
        order = torch.randperm(len(train_bank), generator=generator)
        running = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            features = train_bank.features[rows]
            if train_bank.flipped is not None and flip_probability > 0.0:
                mirror = (
                    torch.rand(len(rows), generator=generator) < flip_probability
                ).unsqueeze(1)
                features = torch.where(mirror, train_bank.flipped[rows], features)
            breakdown = masked_cross_entropy(
                _logits(heads, features.to(device)),
                train_bank.targets[rows].to(device),
                train_bank.supervised[rows].to(device),
                weights,
            )
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            optimizer.step()
            running += float(breakdown.total.detach())
            batches += 1

        scores = head_scores(
            _predict(heads, val_bank, device=device, batch_size=batch_size),
            val_bank.targets,
            val_bank.supervised,
        )
        score = _selection_score(scores)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / batches if batches else 0.0,
                "val_mean_macro_f1": score,
                "val_heads": scores,
            }
        )
        log.info(
            "epoch %d/%d train_loss=%.4f val_mean_macro_f1=%.4f",
            epoch,
            epochs,
            history[-1]["train_loss"],
            score,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in heads.state_dict().items()
            }

        if sidecar is not None:
            paths.ensure_dir(sidecar.parent)
            torch.save(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_state": best_state,
                    "epoch": epoch,
                    "generator": generator.get_state(),
                    "heads": heads.state_dict(),
                    "history": history,
                    "optimizer": optimizer.state_dict(),
                },
                sidecar,
            )

    heads.load_state_dict(best_state)
    return heads, best_epoch, best_score, history


def _split_inputs(names: Sequence[str]) -> dict[str, str]:
    """Return the digest of each named split file that is staged."""
    digests: dict[str, str] = {}
    for name in names:
        candidate = paths.splits_dir() / f"{name}.txt"
        if candidate.is_file():
            digests[name] = paths.file_sha256(candidate)
    return digests


def _split_ids(name: str, limit: int | None) -> list[str]:
    """Return the split members, optionally truncated for a smoke run."""
    ordered = sorted(load_split(name))
    return ordered if limit is None else ordered[:limit]


def train(
    *,
    config_path: Path | None = None,
    table_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
    metrics_out: Path | None = None,
) -> TrainingRun:
    """Extract features once, fit the heads and save the selected checkpoint."""
    config = load_perception_config(
        paths.configs_dir() / "perception_cnn.yaml" if config_path is None else config_path
    )
    settings = _training_settings(config)
    resolved = _resolve_device(device)
    torch.manual_seed(int(settings["seed"]))

    source = table_path or paths.data_root() / "processed" / "outfits.parquet"
    frame = pd.read_parquet(source)

    declared = Path(config["checkpoint"]["path"]) if checkpoint_path is None else checkpoint_path
    checkpoint = declared if declared.is_absolute() else paths.repo_root() / declared

    backbone, feature_dim = build_backbone(str(settings["backbone"]), pretrained=True)
    transform = build_transform(config)
    banks: dict[str, FeatureBank] = {}
    for name in ("train", "val"):
        ids = _split_ids(name, limit)
        dataset = split_dataset(name, frame, image_ids=ids, transform=transform)
        log.info("extracting %s features for %d photographs", name, len(dataset))
        banks[name] = extract_features(
            backbone,
            dataset,
            device=resolved,
            batch_size=int(settings["extract_batch_size"]),
            num_workers=int(settings["num_workers"]),
            with_flip=name == "train",
        )

    weights = class_weights(
        banks["train"].targets.numpy(), banks["train"].supervised.numpy()
    )
    untrained = untrained_heads(weights)
    if untrained:
        log.warning("no supervised training rows for %s", ", ".join(untrained))

    heads, best_epoch, best_score, history = fit_heads(
        banks["train"],
        banks["val"],
        weights,
        device=resolved,
        settings=settings,
        flip_probability=float(config["preprocessing"]["horizontal_flip"]),
        checkpoint=checkpoint,
    )

    model = MultiHeadCNN(backbone.cpu(), feature_dim)
    model.heads.load_state_dict({key: value.cpu() for key, value in heads.state_dict().items()})
    save_checkpoint(
        checkpoint,
        model,
        backbone=str(settings["backbone"]),
        preprocessing=config["preprocessing"],
        class_weights=weights,
        untrained=untrained,
    )
    log.info("wrote checkpoint to %s after epoch %d", checkpoint, best_epoch)

    scores = head_scores(
        _predict(
            heads.to(resolved),
            banks["val"],
            device=resolved,
            batch_size=int(settings["batch_size"]),
        ),
        banks["val"].targets,
        banks["val"].supervised,
    )
    destination = metrics_out or paths.metrics_dir() / "perception_cnn_train.json"
    write_metric_record(
        metric_record(
            stage="perception",
            model=str(settings["backbone"]),
            split="val",
            n_items=len(banks["val"]),
            metrics={
                "best_epoch": best_epoch,
                "class_maps": {head: list(values) for head, values in CLASS_MAPS.items()},
                "history": history,
                "mean_macro_f1": best_score,
                "per_head": scores,
                "untrained_heads": list(untrained),
            },
            config={"perception_cnn": dict(config), "training": settings},
            seed=int(settings["seed"]),
            inputs=_split_inputs(("train", "val")),
        ),
        destination,
    )
    log.info("wrote training metrics to %s", destination)

    resume = resume_path(checkpoint)
    resume.unlink(missing_ok=True)

    return TrainingRun(
        checkpoint=checkpoint,
        best_epoch=best_epoch,
        best_score=best_score,
        history=tuple(history),
        scores=scores,
        untrained=untrained,
    )


def load_split_ids(name: str, limit: int | None) -> list[str]:
    """Return the split members, optionally truncated for a smoke run."""
    from rmo.splits import load_split

    ordered = sorted(load_split(name))
    return ordered if limit is None else ordered[:limit]


def main(argv: Sequence[str] | None = None) -> int:
    """Train the attribute heads and write the selected checkpoint."""
    parser = argparse.ArgumentParser(description="Train the nine attribute heads.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run = train(
        config_path=args.config,
        table_path=args.table,
        checkpoint_path=args.checkpoint,
        device=args.device,
        limit=args.limit,
        metrics_out=args.metrics_out,
    )
    log.info(
        "best epoch %d with mean macro f1 %.4f", run.best_epoch, run.best_score
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
