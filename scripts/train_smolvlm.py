import json
from pathlib import Path

import torch
from datasets import Dataset, Image
from peft import LoraConfig
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from rmo.data.descriptions import load_outfit_table
from rmo.splits import load_split


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("models/smolvlm-smoke")
TRAIN_LIMIT = 32


def make_answer(row) -> str:
    garments = []

    garments.append(
        {
            "slot": "upper",
            "fabric": row["upper_fabric"],
            "pattern": row["upper_pattern"],
            "sleeve_length": row["sleeve_length"],
            "neckline": row["neckline"],
        }
    )

    garments.append(
        {
            "slot": "lower",
            "fabric": row["lower_fabric"],
            "pattern": row["lower_pattern"],
            "length": row["lower_length"],
        }
    )

    if row["outer_fabric"] != "na" or row["outer_pattern"] != "na":
        garments.append(
            {
                "slot": "outer",
                "fabric": row["outer_fabric"],
                "pattern": row["outer_pattern"],
            }
        )

    return json.dumps({"garments": garments})


def make_dataset(limit: int) -> Dataset:
    table = load_outfit_table()
    train_ids = sorted(load_split("train"))

    image_paths = []
    messages = []

    for image_id in train_ids:
        row = table.loc[image_id]

        if not row["has_shape"]:
            continue

        image_path = RAW_DIR / "images" / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        image_paths.append(str(image_path))

        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Describe the visible garments. "
                                "Return their slot, fabric, pattern, "
                                "sleeve length, length and neckline."
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": make_answer(row),
                        }
                    ],
                },
            ]
        )

        if len(image_paths) >= limit:
            break

    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            "messages": messages,
        }
    )

    return dataset.cast_column("image", Image())


dataset = make_dataset(TRAIN_LIMIT)

print(f"Training examples: {len(dataset)}")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
)

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    bf16=True,
    gradient_checkpointing=True,
    max_length=None,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
)

trainer.train()
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))