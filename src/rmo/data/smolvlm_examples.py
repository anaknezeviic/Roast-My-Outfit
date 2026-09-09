"""Build supervised SmolVLM examples from the canonical outfit description.

This module deliberately does not parse the raw DeepFashion annotation files a
second time.  The canonical :mod:`rmo.data.descriptions` layer owns garment
presence, regional-label resolution, shape placement and colour extraction;
SmolVLM training only serialises that already validated representation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rmo.config import load_perception_config
from rmo.data.descriptions import describe_image, describe_split, load_outfit_table
from rmo.data.label_resolution import ONE_PIECE_SLOTS, resolve_regional_value
from rmo.data.preflight import photo_path
from rmo.perception.structured_output import serialize_description
from rmo.schemas import GarmentSlot, OutfitDescription

__all__ = [
    "SMOLVLM_PROMPT_V1",
    "SmolVLMTrainingExample",
    "build_training_example",
    "iter_training_examples",
    "training_omissions",
    "smolvlm_prompt",
]


def smolvlm_prompt(config_path: Path | None = None) -> str:
    """Return the perception prompt shared by training and runtime inference."""
    prompt = load_perception_config(config_path).get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Perception configuration must define a non-empty 'prompt'.")
    return prompt.strip()


# Backward-compatible name for callers/tests that want the default contract.
# The source of truth is configs/perception.yaml; builders with a custom config
# resolve that config dynamically instead of using this default snapshot.
SMOLVLM_PROMPT_V1 = smolvlm_prompt()

# DeepFashion supplies regional fabric/pattern supervision only for these slots.
# Accessories receive colour from the parsing mask, but their fabric/pattern are
# absent from the source annotations; serialising those schema defaults as `na`
# would falsely teach that the annotation said "unknown".
_TEXTURE_SUPERVISED_SLOTS: frozenset[GarmentSlot] = frozenset(
    {
        GarmentSlot.upper,
        GarmentSlot.outer,
        GarmentSlot.lower,
        GarmentSlot.dress,
        GarmentSlot.romper,
    }
)


@dataclass(frozen=True, slots=True)
class SmolVLMTrainingExample:
    """One deterministic image/prompt/completion record before HF conversion."""

    image_id: str
    image_path: Path
    prompt: str
    answer: str

    def as_hf_record(self) -> dict[str, object]:
        """Return TRL's conversational prompt-completion VLM record."""
        return {
            "image": str(self.image_path),
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.answer}],
                }
            ],
        }


def training_omissions(
    description: OutfitDescription,
    row: Mapping[str, object],
) -> dict[str, frozenset[str]]:
    """Return fields that must not become supervised labels for this example.

    There are two reasons to omit a field rather than write ``field=na``:

    * DeepFashion does not annotate accessory fabric/pattern at all.
    * A dress/romper can have disagreeing upper/lower regional labels while the
      RMO schema stores one value per physical garment.  Choosing either side or
      supervising ``na`` would fabricate a target.
    """
    omitted: dict[str, frozenset[str]] = {}
    for garment in description.garments:
        fields: set[str] = set()
        if garment.slot not in _TEXTURE_SUPERVISED_SLOTS:
            fields.update(("fabric", "pattern"))
        elif garment.slot in ONE_PIECE_SLOTS:
            for attribute in ("fabric", "pattern"):
                resolution = resolve_regional_value(
                    row[f"upper_{attribute}"], row[f"lower_{attribute}"]
                )
                if resolution.conflicted:
                    fields.add(attribute)
        if fields:
            omitted[garment.ref] = frozenset(fields)
    return omitted


def _example_from_description(
    description: OutfitDescription,
    row: Mapping[str, object],
    *,
    config_path: Path | None = None,
) -> SmolVLMTrainingExample:
    """Serialise one canonical description without inventing missing labels."""
    answer = serialize_description(
        description,
        omitted_fields=training_omissions(description, row),
    )
    if not answer.strip():
        raise ValueError(f"Canonical description for {description.image_id!r} is empty.")
    return SmolVLMTrainingExample(
        image_id=description.image_id,
        image_path=photo_path(description.image_id),
        prompt=smolvlm_prompt(config_path),
        answer=answer,
    )


def build_training_example(
    image_id: str,
    row: pd.Series,
    *,
    config_path: Path | None = None,
) -> SmolVLMTrainingExample:
    """Build one supervised example through the canonical ground-truth path."""
    description = describe_image(image_id, row, config_path=config_path)
    return _example_from_description(description, row, config_path=config_path)


def iter_training_examples(
    split: str,
    *,
    limit: int | None = None,
    config_path: Path | None = None,
    table: pd.DataFrame | None = None,
) -> Iterator[SmolVLMTrainingExample]:
    """Yield every requested split example; invalid inputs fail instead of skipping."""
    frame = load_outfit_table() if table is None else table
    for description in describe_split(
        split,
        limit=limit,
        config_path=config_path,
        table=frame,
    ):
        yield _example_from_description(
            description,
            frame.loc[description.image_id],
            config_path=config_path,
        )
