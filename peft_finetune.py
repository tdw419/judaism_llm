#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B on Sefaria using PEFT + transformers
Space-efficient version without Unsloth (slower, ~8GB less disk)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# 4-bit quantization config (saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading Qwen2.5-7B model...")
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

print("Preparing model for LoRA...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen2.5 uses explicit attention and MLP module names
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

print("Loading training data...")
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
print(f"Dataset size: {len(dataset)} examples")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False
   )

print("Tokenizing data...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,  # Reduced from 2 to avoid OOM
    gradient_accumulation_steps=16,  # Increased from 8 to keep effective batch size
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    max_steps=1000,
    gradient_checkpointing=True,
    report_to="none",
    # Memory optimizations
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    logging_first_step=True
)

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

print("Saving model...")
model.save_pretrained("judaism-llm-qwen2.5-7b")
tokenizer.save_pretrained("judaism-llm-qwen2.5-7b")

print("Training complete!")
print(f"Model saved to judaism-llm-qwen2.5-7b/")
print(f"LoRA adapters: {list(model.get_adapter_state_dict().keys())}")