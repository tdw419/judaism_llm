#!/usr/bin/env python3
"""
Create a merged model for GGUF export and testing
This script merges LoRA adapters into the base model for inference/export
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "judaism-llm-qwen2.5-7b"
output_path = "judaism-llm-qwen2.5-7b-merged"

print(f"Loading base model: {base_model_name}...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapters from: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapters...")
model = model.merge_and_unload()

print(f"Saving merged model to: {output_path}")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Output directory: {output_path}/")

# Verify saved files
print("\nSaved files:")
for f in os.listdir(output_path):
    size = os.path.getsize(os.path.join(output_path, f))
    print(f"  {f}: {size/1024/1024:.1f} MB")