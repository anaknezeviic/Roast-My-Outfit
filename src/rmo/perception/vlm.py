"""Run local zero-shot SmolVLM perception with lazy model imports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rmo import paths
from rmo.config import load_perception_config
from rmo.imaging import ImageInput, load_image
from rmo.perception.base import PerceptionModel
from rmo.perception.postprocess import parse_description
from rmo.schemas import ColorName, Garment, GarmentSlot, OutfitDescription
from rmo.scoring.palette import PaletteEntry, extract_palette

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_MODEL_ID", "SmolVLMPerception"]

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

_SLOT_PRIORITY: tuple[GarmentSlot, ...] = (
    GarmentSlot.dress,
    GarmentSlot.romper,
    GarmentSlot.outer,
    GarmentSlot.lower,
    GarmentSlot.upper,
    GarmentSlot.footwear,
    GarmentSlot.bag,
    GarmentSlot.headwear,
    GarmentSlot.neckwear,
    GarmentSlot.socks,
    GarmentSlot.belt,
    GarmentSlot.gloves,
    GarmentSlot.eyewear,
    GarmentSlot.jewelry,
    GarmentSlot.other,
)


def _image_identity(image: ImageInput) -> tuple[str, str]:
    """Return the image id and path for an input, empty path for in-memory images."""
    if isinstance(image, (str, Path)):
        return Path(image).stem, str(image)
    return "in_memory", ""


def _keep_labels(array: np.ndarray, labels: frozenset[int]) -> np.ndarray:
    """Return ``array`` with every label outside ``labels`` set to zero."""
    return np.where(np.isin(array, list(labels)), array, 0)


def _measure(garment: Garment, entry: PaletteEntry, area_fraction: float | None) -> None:
    """Record a measured palette entry on a garment, keeping any colour the model stated."""
    garment.color_lab = entry.lab
    garment.color_lab_source = entry.source
    garment.area_fraction = area_fraction
    if garment.color is ColorName.unknown:
        garment.color = entry.name


def _load_mask(image_id: str, size: tuple[int, int], keep: frozenset[int]) -> np.ndarray | None:
    """Return the parsing label map for ``image_id``, or ``None`` when it is unusable."""
    directory = paths.parsing_dir()
    path = next(
        (
            candidate
            for candidate in (directory / f"{image_id}_segm.png", directory / f"{image_id}.png")
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        return None

    with Image.open(path) as handle:
        array = np.asarray(handle)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != size:
        log.warning("parsing mask for %s is %s, expected %s", image_id, array.shape, size)
        return None

    selected = _keep_labels(array, keep)
    if not selected.any():
        return None
    return selected


def _slot_labels(config: dict[str, Any]) -> dict[GarmentSlot, frozenset[int]]:
    """Return the parsing label ids that belong to each garment slot."""
    mask_labels = config.get("mask_labels", {})
    return {slot: frozenset(mask_labels.get(slot.value, ())) for slot in GarmentSlot}


def _apply_palette(
    garments: list[Garment],
    image: Image.Image,
    mask: np.ndarray | None,
    labels: dict[GarmentSlot, frozenset[int]],
) -> None:
    """Fill measured colour on each garment from the parsing mask, or the whole image."""
    if mask is None:
        entries = extract_palette(image, None, n_colors=1)
        if not entries:
            return
        target = min(
            enumerate(garments),
            key=lambda item: (_SLOT_PRIORITY.index(item[1].slot), item[0]),
        )[1]
        _measure(target, entries[0], None)
        return

    denominator = int(np.count_nonzero(mask))
    for garment in garments:
        wanted = labels[garment.slot]
        if not wanted:
            continue
        selected = _keep_labels(mask, wanted)
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        entries = extract_palette(image, selected, n_colors=1)
        if not entries:
            continue
        _measure(garment, entries[0], count / denominator)


class SmolVLMPerception(PerceptionModel):
    """Zero-shot outfit perception with a local SmolVLM checkpoint."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Record the checkpoint and prompt without loading any weights."""
        config = load_perception_config(config_path)
        self.name = model_id
        self._config_path = config_path
        self._prompt: str = config["prompt"]
        self._generation: dict[str, Any] = dict(config["generation"])
        self._labels = _slot_labels(config)
        self._requested_device = device
        self._device: str | None = None
        self._processor: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Load the processor and model on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SmolVLMPerception needs torch and transformers; install the vlm extra."
            ) from exc

        device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.name)
        self._model = AutoModelForImageTextToText.from_pretrained(self.name, dtype=dtype).to(
            device
        )
        self._model.eval()
        self._device = device
        log.info("loaded %s on %s as %s", self.name, device, dtype)

    def _generate(self, image: Image.Image) -> str:
        """Return the raw text the model produces for one image."""
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self._prompt}]}
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **self._generation)
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def predict(self, image: ImageInput) -> OutfitDescription:
        """Return the description of the outfit in one image."""
        picture = load_image(image)
        image_id, image_path = _image_identity(image)
        raw = self._generate(picture)
        description = parse_description(
            raw,
            image_id=image_id,
            source_model=self.name,
            image_path=image_path,
            config_path=self._config_path,
        )
        keep = frozenset().union(*self._labels.values())
        mask = _load_mask(image_id, (picture.height, picture.width), keep)
        _apply_palette(description.garments, picture, mask, self._labels)
        return description
