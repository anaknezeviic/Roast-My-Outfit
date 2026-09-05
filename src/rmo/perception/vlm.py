"""Run local zero-shot SmolVLM perception with lazy model imports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from rmo import paths
from rmo.config import load_perception_config
from rmo.imaging import ImageInput, image_identity, load_image
from rmo.perception.base import PerceptionModel
from rmo.perception.enrichment import apply_palette, load_mask, slot_labels
from rmo.perception.postprocess import parse_description
from rmo.schemas import OutfitDescription

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_MODEL_ID", "REGISTRY_NAME", "SmolVLMPerception"]

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

REGISTRY_NAME = "smolvlm"


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
        self._labels = slot_labels(config)
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
        image_id, image_path = image_identity(image)
        picture = load_image(image)
        raw = self._generate(picture)
        description = parse_description(
            raw,
            image_id=image_id,
            source_model=self.name,
            image_path=paths.export_path(image_path),
            config_path=self._config_path,
        )
        keep = frozenset().union(*self._labels.values())
        mask = load_mask(image_id, (picture.height, picture.width), keep)
        apply_palette(description.garments, picture, mask, self._labels)
        return description
