"""Nine-head frozen-backbone attribute model, its dataset and its masked loss."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

from rmo import paths
from rmo.imaging import load_image
from rmo.schemas import Fabric, LowerLength, Neckline, Pattern, SleeveLength
from rmo.splits import load_split

log = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CLASS_MAPS",
    "DEFAULT_PREPROCESSING",
    "HEAD_CLASS_COUNTS",
    "HEAD_NAMES",
    "HEAD_VOCABULARIES",
    "SHAPE_HEADS",
    "CheckpointError",
    "LossBreakdown",
    "MultiHeadCNN",
    "OutfitAttributeDataset",
    "build_backbone",
    "build_transform",
    "class_weights",
    "encode_frame",
    "encode_label",
    "load_checkpoint",
    "masked_cross_entropy",
    "save_checkpoint",
    "split_dataset",
    "untrained_heads",
]

HEAD_VOCABULARIES: dict[str, type[Enum]] = {
    "upper_fabric": Fabric,
    "lower_fabric": Fabric,
    "outer_fabric": Fabric,
    "upper_pattern": Pattern,
    "lower_pattern": Pattern,
    "outer_pattern": Pattern,
    "sleeve_length": SleeveLength,
    "lower_length": LowerLength,
    "neckline": Neckline,
}

HEAD_NAMES: tuple[str, ...] = tuple(HEAD_VOCABULARIES)

SHAPE_HEADS: frozenset[str] = frozenset({"sleeve_length", "lower_length", "neckline"})

CLASS_MAPS: dict[str, tuple[str, ...]] = {
    head: tuple(member.value for member in vocabulary)
    for head, vocabulary in HEAD_VOCABULARIES.items()
}

HEAD_CLASS_COUNTS: dict[str, int] = {head: len(values) for head, values in CLASS_MAPS.items()}

DEFAULT_PREPROCESSING: dict[str, Any] = {
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "horizontal_flip": 0.5,
}

CHECKPOINT_FORMAT = "rmo_cnn_multihead"
CHECKPOINT_VERSION = 1

_PAYLOAD_KEYS: tuple[str, ...] = (
    "format",
    "version",
    "backbone",
    "feature_dim",
    "class_maps",
    "preprocessing",
    "class_weights",
    "untrained_heads",
    "state_dict",
)


class CheckpointError(ValueError):
    """Raised when a checkpoint does not match the current model contract."""


def encode_label(head: str, value: str) -> int:
    """Return the class index ``value`` occupies in the vocabulary of ``head``."""
    try:
        vocabulary = CLASS_MAPS[head]
    except KeyError:
        raise ValueError(f"Unknown head {head!r}; expected one of {', '.join(HEAD_NAMES)}.") from None
    try:
        return vocabulary.index(value)
    except ValueError:
        raise ValueError(f"Head {head!r} has no class {value!r}.") from None


def encode_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return per-head class indices and supervision flags for every row of ``frame``."""
    missing = [column for column in (*HEAD_NAMES, "has_shape") if column not in frame.columns]
    if missing:
        raise ValueError(f"Outfit table is missing columns: {', '.join(missing)}.")

    has_shape = frame["has_shape"].to_numpy(dtype=bool)
    targets = np.zeros((len(frame), len(HEAD_NAMES)), dtype=np.int64)
    supervised = np.ones((len(frame), len(HEAD_NAMES)), dtype=bool)
    for index, head in enumerate(HEAD_NAMES):
        targets[:, index] = [encode_label(head, value) for value in frame[head]]
        if head in SHAPE_HEADS:
            supervised[:, index] = has_shape
    return targets, supervised


def class_weights(targets: np.ndarray, supervised: np.ndarray) -> dict[str, list[float]]:
    """Return inverse-frequency class weights per head over its supervised rows."""
    weights: dict[str, list[float]] = {}
    for index, head in enumerate(HEAD_NAMES):
        size = HEAD_CLASS_COUNTS[head]
        counts = np.bincount(targets[supervised[:, index], index], minlength=size).astype(float)
        total = float(counts.sum())
        present = int(np.count_nonzero(counts))
        if present == 0:
            log.warning("head %s has no supervised rows, so every class weight is zero", head)
            weights[head] = [0.0] * size
            continue
        weights[head] = [total / (present * count) if count else 0.0 for count in counts]
    return weights


def untrained_heads(weights: Mapping[str, Sequence[float]]) -> tuple[str, ...]:
    """Return the heads whose every class weight is zero."""
    return tuple(head for head, values in weights.items() if not any(values))


def build_backbone(name: str, *, pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return a pooled timm feature extractor and the width of its output."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            f"Building backbone {name!r} needs timm; install the cnn extra."
        ) from exc
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, int(model.num_features)


def build_transform(
    config: Mapping[str, Any], *, train: bool = False
) -> Callable[[Image.Image], Tensor]:
    """Return the preprocessing pipeline the configuration describes."""
    preprocessing = config["preprocessing"]
    size = int(preprocessing["image_size"])
    steps: list[Callable[..., Any]] = [v2.Resize((size, size))]
    if train:
        steps.append(v2.RandomHorizontalFlip(p=float(preprocessing["horizontal_flip"])))
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(preprocessing["mean"]), std=list(preprocessing["std"])),
    ]
    return v2.Compose(steps)


class MultiHeadCNN(nn.Module):
    """Frozen feature backbone with one linear head per attribute."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        class_counts: Mapping[str, int] | None = None,
    ) -> None:
        """Attach the heads and freeze every backbone parameter."""
        super().__init__()
        counts = dict(HEAD_CLASS_COUNTS if class_counts is None else class_counts)
        surplus = sorted(set(counts) - set(HEAD_NAMES))
        absent = [head for head in HEAD_NAMES if head not in counts]
        if surplus or absent:
            raise ValueError(
                f"Head map must name exactly the nine heads; "
                f"missing {', '.join(absent) or 'nothing'}, "
                f"surplus {', '.join(surplus) or 'nothing'}."
            )
        empty = [head for head in HEAD_NAMES if counts[head] < 1]
        if empty:
            raise ValueError(f"Heads {', '.join(empty)} need at least one class.")

        self.feature_dim = int(feature_dim)
        self.class_counts = {head: int(counts[head]) for head in HEAD_NAMES}
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {
                head: nn.Linear(self.feature_dim, self.class_counts[head])
                for head in HEAD_NAMES
            }
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def forward(self, pixels: Tensor) -> dict[str, Tensor]:
        """Return one logit tensor per head, in head declaration order."""
        with torch.no_grad():
            features = self.backbone(pixels)
        return {head: self.heads[head](features) for head in HEAD_NAMES}

    def train(self, mode: bool = True) -> MultiHeadCNN:
        """Switch the heads into ``mode`` while the backbone stays in evaluation mode."""
        super().train(mode)
        # a training backbone would update its normalisation buffers and drop activations
        self.backbone.eval()
        return self

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters an optimizer is allowed to update."""
        return self.heads.parameters()


class OutfitAttributeDataset(Dataset):
    """One full-body outfit row per item, with per-head supervision masks."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_ids: Sequence[str],
        *,
        image_dir: Path | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        """Resolve every requested row and photograph before any batch is drawn."""
        ordered = sorted(image_ids)
        repeated = sorted(name for name, count in Counter(ordered).items() if count > 1)
        if repeated:
            raise ValueError(f"Image id {repeated[0]!r} was requested more than once.")

        for column in ("image_id", "is_full_body"):
            if column not in frame.columns:
                raise ValueError(f"Outfit table is missing columns: {column}.")

        rows: dict[str, list[int]] = {}
        for position, image_id in enumerate(frame["image_id"]):
            rows.setdefault(str(image_id), []).append(position)
        for image_id in ordered:
            found = rows.get(image_id, [])
            if not found:
                raise ValueError(f"Image id {image_id!r} is absent from the outfit table.")
            if len(found) > 1:
                raise ValueError(
                    f"Image id {image_id!r} matches {len(found)} rows of the outfit table."
                )

        selected = frame.iloc[[rows[image_id][0] for image_id in ordered]]
        partial = [
            image_id
            for image_id, full_body in zip(ordered, selected["is_full_body"], strict=True)
            if not bool(full_body)
        ]
        if partial:
            raise ValueError(f"Image id {partial[0]!r} is not a full-body photograph.")

        directory = paths.raw_dir() / "images" if image_dir is None else Path(image_dir)
        photos = [directory / f"{image_id}.jpg" for image_id in ordered]
        absent = next((photo for photo in photos if not photo.is_file()), None)
        if absent is not None:
            raise ValueError(f"No photograph at {absent}.")

        self.image_ids: tuple[str, ...] = tuple(ordered)
        self.targets, self.supervised = encode_frame(selected)
        self._photos = tuple(photos)
        self._transform = (
            build_transform({"preprocessing": DEFAULT_PREPROCESSING})
            if transform is None
            else transform
        )

    def __len__(self) -> int:
        """Return how many photographs the dataset holds."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the pixels, targets and supervision flags of one photograph."""
        return {
            "image_id": self.image_ids[index],
            "pixels": self._transform(load_image(self._photos[index])),
            "targets": torch.from_numpy(self.targets[index].copy()),
            "supervised": torch.from_numpy(self.supervised[index].copy()),
        }


def split_dataset(
    name: str,
    frame: pd.DataFrame,
    *,
    image_ids: Iterable[str] | None = None,
    image_dir: Path | None = None,
    transform: Callable[[Image.Image], Tensor] | None = None,
) -> OutfitAttributeDataset:
    """Return the dataset for one frozen split, or for a subset of it."""
    available = load_split(name)
    if image_ids is None:
        requested: list[str] = sorted(available)
    else:
        requested = list(image_ids)
        foreign = next((image_id for image_id in requested if image_id not in available), None)
        if foreign is not None:
            raise ValueError(f"Image id {foreign!r} is not part of the {name} split.")
    return OutfitAttributeDataset(
        frame, requested, image_dir=image_dir, transform=transform
    )


@dataclass(frozen=True)
class LossBreakdown:
    """Per-head losses and the heads that contributed to the average."""

    total: Tensor
    per_head: dict[str, Tensor]
    active: tuple[str, ...]


def masked_cross_entropy(
    logits: Mapping[str, Tensor],
    targets: Tensor,
    supervised: Tensor,
    weights: Mapping[str, Sequence[float]],
) -> LossBreakdown:
    """Return the class-weighted cross-entropy of every head that has supervised rows."""
    per_head: dict[str, Tensor] = {}
    active: list[str] = []
    for index, head in enumerate(HEAD_NAMES):
        head_logits = logits[head]
        rows = supervised[:, index]
        labels = targets[:, index][rows]
        table = torch.as_tensor(
            list(weights[head]), dtype=head_logits.dtype, device=head_logits.device
        )
        row_weights = table[labels]
        denominator = row_weights.sum()
        if float(denominator) <= 0.0:
            per_head[head] = head_logits.sum() * 0.0
            continue
        losses = F.cross_entropy(head_logits[rows], labels, reduction="none")
        per_head[head] = (row_weights * losses).sum() / denominator
        active.append(head)

    if active:
        total = torch.stack([per_head[head] for head in active]).mean()
    else:
        total = torch.stack([logits[head].sum() * 0.0 for head in HEAD_NAMES]).sum()
    return LossBreakdown(total=total, per_head=per_head, active=tuple(active))


def save_checkpoint(
    path: Path,
    model: MultiHeadCNN,
    *,
    backbone: str,
    preprocessing: Mapping[str, Any],
    class_weights: Mapping[str, Sequence[float]] | None = None,
    untrained: Sequence[str] = (),
) -> None:
    """Write the model weights beside the contract they were trained under."""
    if model.class_counts != HEAD_CLASS_COUNTS:
        raise CheckpointError(
            "Only a model with the standard head cardinalities can be saved; "
            f"got {model.class_counts}, expected {HEAD_CLASS_COUNTS}."
        )
    destination = Path(path)
    paths.ensure_dir(destination.parent)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "backbone": backbone,
        "feature_dim": model.feature_dim,
        "class_maps": {head: list(values) for head, values in CLASS_MAPS.items()},
        "preprocessing": dict(preprocessing),
        "class_weights": {
            head: list(values) for head, values in (class_weights or {}).items()
        },
        "untrained_heads": list(untrained),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, destination)


def load_checkpoint(
    path: Path, *, preprocessing: Mapping[str, Any] | None = None
) -> tuple[MultiHeadCNN, dict[str, Any]]:
    """Rebuild the model a checkpoint describes and return it with its metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"No checkpoint at {source}.")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    missing = [key for key in _PAYLOAD_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"Checkpoint {source} is missing the keys {', '.join(missing)}.")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"Checkpoint {source} is in format {payload['format']!r}, not {CHECKPOINT_FORMAT!r}."
        )
    if payload["version"] > CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Checkpoint {source} is version {payload['version']}, "
            f"newer than the supported {CHECKPOINT_VERSION}."
        )
    stored = {head: tuple(values) for head, values in payload["class_maps"].items()}
    if stored != CLASS_MAPS:
        raise CheckpointError(f"Checkpoint {source} was trained on a different class vocabulary.")
    if preprocessing is not None and dict(preprocessing) != dict(payload["preprocessing"]):
        raise CheckpointError(f"Checkpoint {source} was trained with different preprocessing.")

    backbone, feature_dim = build_backbone(payload["backbone"])
    if feature_dim != payload["feature_dim"]:
        raise CheckpointError(
            f"Backbone {payload['backbone']!r} is {feature_dim} wide, "
            f"but checkpoint {source} expects {payload['feature_dim']}."
        )
    model = MultiHeadCNN(backbone, feature_dim)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint {source} does not fit the model: {exc}.") from exc

    metadata = {key: payload[key] for key in _PAYLOAD_KEYS if key != "state_dict"}
    return model.eval(), metadata
