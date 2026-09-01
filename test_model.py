#!/usr/bin/env python3
"""
Test the trained Judaism LLM by loading LoRA adapters and generating text
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

# Load base model with 4-bit quantization
model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "judaism-llm-qwen2.5-7b"

print(f"Loading base model: {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print(f"Loading LoRA adapters from: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

print("Testing generation...")

# Test prompts
test_prompts = [
    "What is the significance of Shabbat in Jewish tradition?",
    "Explain the concept of Teshuva.",
    "What are the main sources of Jewish law?",
    "תרגם: Blessed are You, Lord our God, King of the universe",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {prompt}")
    print('='*60)

    # Format for instruction-tuned model
    messages = [
        {"role": "system", "content": "You are an expert in Jewish texts including Torah, Talmud, Mishnah, and commentaries. Provide accurate, well-sourced responses in both Hebrew and English."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")

print("\n" + "="*60)
print("Testing complete!")
print(f"LoRA adapter size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
print(f"Total model size: {sum(p.numel() for p in model.parameters()):,} parameters")