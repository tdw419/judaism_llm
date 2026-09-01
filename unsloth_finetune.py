#!/usr/bin/env python3
"""
Unsloth fine-tuning script for Judaism LLM
Fine-tunes Qwen2.5-7B on Sefaria corpus using LoRA adapters.
"""

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading Qwen2.5-7B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Configuring LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Loading training data...")
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
print(f"Dataset size: {len(dataset)} examples")

def formatting_prompts_func(examples):
    return {"text": examples["text"]}

print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=1000,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
    ),
)

trainer.train()

print("Saving model...")
model.save_pretrained("judaism-llm-qwen2.5-7b")
tokenizer.save_pretrained("judaism-llm-qwen2.5-7b")

print("Training complete! Model saved to judaism-llm-qwen2.5-7b/")
print("To merge and export:")
print("  from unsloth import FastLanguageModel")
print("  model, tokenizer = FastLanguageModel.from_pretrained('judaism-llm-qwen2.5-7b', load_in_4bit=False)")
print("  FastLanguageModel.merge_and_save('judaism-llm-qwen2.5-7b-merged')")