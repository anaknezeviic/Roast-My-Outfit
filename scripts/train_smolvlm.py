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
OUTPUT_DIR = Path("models/smolvlm-rmo-v1")
PROMPT = (
    "Describe every visible garment. "
    "For each garment state its slot, pattern, fabric, "
    "sleeve length, garment length and neckline when applicable."
)


def make_answer(row) -> str:
    lines = []

    if row["upper_fabric"] != "na" or row["upper_pattern"] != "na":
        values = [
            row["upper_fabric"],
            row["upper_pattern"],
        ]

        if row["sleeve_length"] != "na":
            values.append(row["sleeve_length"])

        if row["neckline"] != "na":
            values.append(row["neckline"])

        lines.append(f"upper: {', '.join(values)}")

    if row["lower_fabric"] != "na" or row["lower_pattern"] != "na":
        values = [
            row["lower_fabric"],
            row["lower_pattern"],
        ]

        if row["lower_length"] != "na":
            values.append(row["lower_length"])

        lines.append(f"lower: {', '.join(values)}")

    if row["outer_fabric"] != "na" or row["outer_pattern"] != "na":
        values = [
            value
            for value in (
                row["outer_fabric"],
                row["outer_pattern"],
            )
            if value != "na"
        ]

        lines.append(f"outer: {', '.join(values)}")

    return "\n".join(lines)


def make_dataset(split_name: str) -> Dataset:
    table = load_outfit_table()
    image_ids = sorted(load_split(split_name))

    image_paths = []
    prompts = []
    completions = []

    for image_id in image_ids:
        row = table.loc[image_id]

        if not row["has_shape"]:
            continue

        image_path = RAW_DIR / "images" / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        answer = make_answer(row)

        if not answer:
            continue

        image_paths.append(str(image_path))

        prompts.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
        )

        completions.append(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                }
            ]
        )

    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            "prompt": prompts,
            "completion": completions,
        }
    )

    return dataset.cast_column("image", Image())

train_dataset = make_dataset("train")
val_dataset = make_dataset("val")

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

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
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=0.05,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    bf16=True,
    bf16_full_eval=True,
    gradient_checkpointing=True,
    max_length=None,
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=processor,
)

trainer.train()

trainer.save_model(str(OUTPUT_DIR / "final"))
processor.save_pretrained(str(OUTPUT_DIR / "final"))