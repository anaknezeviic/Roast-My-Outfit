"""Nine-head CNN perception behind the shared model interface."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from rmo import paths
from rmo.config import load_perception_config
from rmo.imaging import ImageInput, image_identity, load_image
from rmo.perception.base import PerceptionModel
from rmo.perception.enrichment import (
    HEAD_PROJECTION,
    SLOT_CATEGORIES,
    apply_palette,
    load_mask,
    mask_skeleton,
    slot_labels,
)
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

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

__all__ = ["HeadReadout", "CNNPerception"]

_FIELD_TYPES: dict[str, type] = {
    "fabric": Fabric,
    "pattern": Pattern,
    "sleeve_length": SleeveLength,
    "length": LowerLength,
    "neckline": Neckline,
}

_FALLBACK_HEADS: dict[str, tuple[str, ...]] = {
    "fabric": ("upper_fabric", "lower_fabric", "outer_fabric"),
    "pattern": ("upper_pattern", "lower_pattern", "outer_pattern"),
}

_NA = "na"

_Reading = tuple[dict[str, str], dict[str, float]]


@dataclass(frozen=True, slots=True)
class HeadReadout:
    """Raw per-head decisions for one image, outside the strict schema."""

    image_id: str
    labels: dict[str, str]
    probabilities: dict[str, float]
    mask_classes: dict[str, int]
    unmapped_refs: tuple[str, ...]
    fallback: bool


class CNNPerception(PerceptionModel):
    """Nine-head attribute CNN projected onto a mask-derived garment skeleton."""

    name = "cnn_multihead_v1"

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        """Read the configuration and locate the checkpoint without loading any weights."""
        resolved = (
            paths.configs_dir() / "perception_cnn.yaml" if config_path is None else Path(config_path)
        )
        self._config = load_perception_config(resolved)
        self._labels = slot_labels(load_perception_config(None))
        self._batch_size = int(self._config["inference"]["batch_size"])
        candidate = (
            Path(self._config["checkpoint"]["path"])
            if checkpoint_path is None
            else Path(checkpoint_path)
        )
        self._checkpoint_path = (
            candidate if candidate.is_absolute() else paths.repo_root() / candidate
        )
        self._requested_device = device
        self._device: str | None = None
        self._model: Any = None
        self._transform: Any = None
        self._cnn: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        """Import torch, restore the checkpoint and move it onto a device."""
        if self._model is not None:
            return
        try:
            import torch

            from rmo.perception import cnn
        except ImportError as exc:
            raise ImportError(
                "CNNPerception needs torch and timm; install the cnn extra."
            ) from exc

        model, _ = cnn.load_checkpoint(
            self._checkpoint_path, preprocessing=self._config["preprocessing"]
        )
        device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._cnn = cnn
        self._transform = cnn.build_transform(self._config)
        self._model = model.to(device).eval()
        self._device = device
        log.info("loaded %s from %s on %s", self.name, self._checkpoint_path, device)

    def _decode(self, pictures: Sequence[Image.Image]) -> list[_Reading]:
        """Return the winning class and its probability for every head of every picture."""
        torch = self._torch
        batch = torch.stack([self._transform(picture) for picture in pictures]).to(self._device)
        scores: dict[str, list[list[float]]] = {}
        with torch.inference_mode():
            for head, values in self._model(batch).items():
                if not bool(torch.isfinite(values).all()):
                    raise ValueError(f"Head {head} produced a non-finite logit.")
                scores[head] = values.softmax(dim=-1).float().cpu().tolist()

        readings: list[_Reading] = []
        for position in range(len(pictures)):
            labels: dict[str, str] = {}
            probabilities: dict[str, float] = {}
            for head, rows in scores.items():
                row = rows[position]
                index = max(range(len(row)), key=row.__getitem__)
                labels[head] = self._cnn.CLASS_MAPS[head][index]
                probabilities[head] = float(row[index])
            readings.append((labels, probabilities))
        return readings

    def _skeleton_garments(
        self, mask: np.ndarray, labels: dict[str, str]
    ) -> tuple[list[Garment], list[list[str]], list[int]]:
        """Build one garment per claimed slot and project the heads onto them."""
        entries = mask_skeleton(mask, self._labels)
        garments = [
            Garment(slot=slot, category=SLOT_CATEGORIES[slot], confidence=0.0)
            for slot, _ in entries
        ]
        assigned: list[list[str]] = [[] for _ in garments]
        for head, (field, candidates) in HEAD_PROJECTION.items():
            position = next(
                (
                    index
                    for slot in candidates
                    for index, entry in enumerate(entries)
                    if entry[0] is slot
                ),
                None,
            )
            if position is None:
                continue
            setattr(garments[position], field, _FIELD_TYPES[field](labels[head]))
            assigned[position].append(head)
        return garments, assigned, [class_id for _, class_id in entries]

    def _fallback_garments(
        self, labels: dict[str, str]
    ) -> tuple[list[Garment], list[list[str]], list[int]]:
        """Build the single unlocated garment that carries the attribute heads alone."""
        garment = Garment(
            slot=GarmentSlot.other,
            category=SLOT_CATEGORIES[GarmentSlot.other],
            confidence=0.0,
        )
        assigned: list[str] = []
        for field, candidates in _FALLBACK_HEADS.items():
            head = next((name for name in candidates if labels[name] != _NA), None)
            if head is None:
                continue
            setattr(garment, field, _FIELD_TYPES[field](labels[head]))
            assigned.append(head)
        return [garment], [assigned], []

    def _describe(
        self, image: ImageInput, picture: Image.Image, reading: _Reading
    ) -> tuple[OutfitDescription, HeadReadout]:
        """Turn one set of head decisions into a description and its raw readout."""
        labels, probabilities = reading
        image_id, source = image_identity(image)
        keep = frozenset().union(*self._labels.values())
        mask = load_mask(image_id, (picture.height, picture.width), keep)
        if mask is None:
            garments, assigned, classes = self._fallback_garments(labels)
        else:
            garments, assigned, classes = self._skeleton_garments(mask, labels)

        for garment, heads in zip(garments, assigned, strict=True):
            if heads:
                garment.confidence = sum(probabilities[head] for head in heads) / len(heads)

        description = OutfitDescription(
            image_id=image_id,
            image_path=paths.export_path(source),
            garments=garments,
            source_model=self.name,
        )
        apply_palette(description.garments, picture, mask, self._labels)

        readout = HeadReadout(
            image_id=image_id,
            labels=labels,
            probabilities=probabilities,
            mask_classes=(
                {} if mask is None else dict(zip(description.refs(), classes, strict=True))
            ),
            unmapped_refs=tuple(
                garment.ref
                for garment, heads in zip(description.garments, assigned, strict=True)
                if not heads
            ),
            fallback=mask is None,
        )
        return description, readout

    def predict(self, image: ImageInput) -> OutfitDescription:
        """Return the description of the outfit in one image."""
        return self.predict_with_readout(image)[0]

    def predict_batch(self, images: Sequence[ImageInput]) -> list[OutfitDescription]:
        """Return one description per image, in input order."""
        return [description for description, _ in self.predict_batch_with_readouts(images)]

    def predict_with_readout(self, image: ImageInput) -> tuple[OutfitDescription, HeadReadout]:
        """Return the description of one image beside the raw head decisions behind it."""
        return self.predict_batch_with_readouts([image])[0]

    def predict_batch_with_readouts(
        self, images: Sequence[ImageInput]
    ) -> list[tuple[OutfitDescription, HeadReadout]]:
        """Return one description and readout per image, in input order."""
        if not images:
            return []
        self._ensure_loaded()
        results: list[tuple[OutfitDescription, HeadReadout]] = []
        for start in range(0, len(images), self._batch_size):
            chunk = list(images[start : start + self._batch_size])
            pictures = [load_image(image) for image in chunk]
            results += [
                self._describe(image, picture, reading)
                for image, picture, reading in zip(
                    chunk, pictures, self._decode(pictures), strict=True
                )
            ]
        return results
