from pathlib import Path

import torch
from datasets import Dataset, Image
from peft import LoraConfig
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from rmo.data.smolvlm_examples import iter_training_examples


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
OUTPUT_DIR = Path("models/smolvlm-rmo-v1")

def make_dataset(split_name: str) -> Dataset:
    records = [example.as_hf_record() for example in iter_training_examples(split_name)]
    dataset = Dataset.from_list(records)
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