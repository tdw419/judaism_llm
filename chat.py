#!/usr/bin/env python3
"""
CLI chat interface for Judaism LLM
Loads merged model and provides interactive Hebrew/English chat
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import re

# Configuration
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

def load_model():
    """Load the merged model and tokenizer."""
    print(f"Loading model from: {MODEL_PATH}")
    print("This will take 1-2 minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """Generate response from the model."""
    messages = [
        {
            "role": "system",
            "content": "You are Judaism LLM, an AI assistant trained on Sefaria corpus. You specialize in Torah, Talmud, Mishnah, commentaries, and Hebrew-English translation. Provide accurate, well-sourced responses in both Hebrew and English."
        },
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
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

def detect_language(text):
    """Simple Hebrew/English detection."""
    hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
    total_chars = len(text.strip())

    if total_chars == 0:
        return "unknown"

    ratio = hebrew_chars / total_chars

    if ratio > 0.3:
        return "Hebrew"
    else:
        return "English"

def main():
    """Interactive chat loop."""
    print("=" * 70)
    print("            Judaism LLM - Interactive Chat")
    print("=" * 70)
    print("Ask questions about Jewish texts in English or Hebrew.")
    print("Type 'quit', 'exit', or 'סיום' to stop.")
    print()

    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've run: python3 merge_model.py")
        sys.exit(1)

    print("\nReady! Ask your question below.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "סיום"]:
                print("\nShalom! שלום עד מפגש נוסף!")
                break

            lang = detect_language(user_input)
            print(f"\n[{lang}] Generating response...\n")

            response = generate_response(model, tokenizer, user_input)

            # Clean up response
            response = response.strip()
            if response.startswith('["') or response.startswith("['"):
                # Remove potential JSON artifacts
                response = re.sub(r'^["\[\]\',]+', '', response)

            print(response)
            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nShalom! שלום עד מפגש נוסף!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            print("Try rephrasing your question or check the model.")

if __name__ == "__main__":
    main()