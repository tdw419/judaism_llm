#!/usr/bin/env python3
"""
CLI chat interface for Judaism LLM
Loads merged model and provides interactive Hebrew/English chat
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bible_codes"))
from els_search import load_torah, render_matrix, search_all_books  # noqa: E402
from gematria import find_skip_sum, find_verses_with_value, load_numeric_torah, word_value  # noqa: E402
from prompts import build_messages  # noqa: E402

# Configuration
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

ELS_DISCLAIMER = (
    "NOTE: ELS 'Bible code' matches are a statistical artifact of searching "
    "long texts with many free parameters (skip, direction, spelling). McKay, "
    "Bar-Natan, Bar-Hillel & Kalai (Statistical Science, 1999) found equally "
    "'significant' matches in Moby Dick using the same method as the original "
    "Witztum-Rips study. Treat this as a text-search toy, not a prediction tool."
)

GEMATRIA_DISCLAIMER = (
    "NOTE: gematria 'hidden code' searches (verse sums, equidistant sums "
    "matching a target number) are exploratory pattern-matching, not "
    "validated discoveries -- with free choice of target, skip, and run "
    "length, matches are expected by chance in any long text. Traditional "
    "gematria word-values themselves are a real, long-used feature of "
    "Jewish textual tradition; 'hidden codes' built on top of them are not."
)

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

def generate_from_messages(model, tokenizer, messages):
    """Like generate_response but takes pre-built chat messages (for RAG)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def _load_els_collection():
    """Lazily load the local ChromaDB RAG collection, or None if unavailable."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from retrieval import retrieve

        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection("sefaria_texts")
        embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return collection, embedding_model, retrieve
    except Exception as e:
        print(f"(commentary lookup unavailable: {e})")
        return None, None, None


def handle_els_command(command, model, tokenizer, torah_cache):
    """Handle '/elscode <hebrew word> [min_skip] [max_skip]'."""
    parts = command.split()
    if len(parts) < 2:
        print("Usage: /elscode <hebrew word> [min_skip] [max_skip]")
        return torah_cache

    word = parts[1]
    min_skip = int(parts[2]) if len(parts) > 2 else -100
    max_skip = int(parts[3]) if len(parts) > 3 else 100

    if torah_cache is None:
        print("Loading Torah text for ELS search (first run only)...")
        try:
            torah_cache = load_torah()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return torah_cache

    print(f"\n{ELS_DISCLAIMER}\n")
    matches = search_all_books(torah_cache, word, min_skip, max_skip)
    print(f"Found {len(matches)} match(es) for '{word}' across skips [{min_skip}, {max_skip}]\n")

    for m in matches[:5]:
        print(f"  {m.book} skip={m.skip:+d}  {m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]}")

    if not matches:
        return torah_cache

    top = matches[0]
    print("\n" + render_matrix(torah_cache[top.book], top) + "\n")

    collection, embedding_model, retrieve = _load_els_collection()
    if collection is None:
        return torah_cache

    print(f"\nLooking up commentary near {top.book} {top.start_ref[0]}:{top.start_ref[1]}...\n")
    query = f"{top.book} {top.start_ref[0]}:{top.start_ref[1]} {word}"
    r = retrieve(query, collection, embedding_model, top_k=3)
    docs = r["documents"][0] if r.get("documents") else []
    metas = r["metadatas"][0] if r.get("metadatas") else []
    if not docs:
        print("No related commentary found in the RAG corpus.")
        return torah_cache

    messages = build_messages(
        f"An ELS (skip={top.skip}) match for the word '{word}' was found starting near "
        f"{top.book} {top.start_ref[0]}:{top.start_ref[1]}. Using only the passages below, "
        f"what does classical commentary say about this verse?",
        docs, metas,
    )
    if model is None or tokenizer is None:
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            src = meta.get("source", "Unknown")
            print(f"  [{i}] ({src}): {doc[:160]}...\n")
    else:
        print(generate_from_messages(model, tokenizer, messages).strip())
    print("-" * 70 + "\n")
    return torah_cache


def handle_gematria_command(command, numeric_torah_cache):
    """Handle '/gematria word|verse-sum|skip-sum ...' subcommands.

    Usage:
      /gematria word <hebrew word>
      /gematria verse-sum <target> [limit]
      /gematria skip-sum <target> [length] [min_skip] [max_skip] [limit]
    """
    parts = command.split()
    if len(parts) < 2:
        print("Usage: /gematria word <word> | verse-sum <target> | skip-sum <target> [length] [min_skip] [max_skip]")
        return numeric_torah_cache

    subcmd = parts[1]

    if subcmd == "word":
        if len(parts) < 3:
            print("Usage: /gematria word <hebrew word>")
            return numeric_torah_cache
        print(f"\n{parts[2]} = {word_value(parts[2])}\n")
        return numeric_torah_cache

    if subcmd not in ("verse-sum", "skip-sum"):
        print("Unknown /gematria subcommand. Use: word | verse-sum | skip-sum")
        return numeric_torah_cache

    if numeric_torah_cache is None:
        print("Loading Torah gematria values (first run only)...")
        try:
            numeric_torah_cache = load_numeric_torah()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return numeric_torah_cache

    print(f"\n{GEMATRIA_DISCLAIMER}\n")

    if subcmd == "verse-sum":
        if len(parts) < 3:
            print("Usage: /gematria verse-sum <target> [limit]")
            return numeric_torah_cache
        target = int(parts[2])
        limit = int(parts[3]) if len(parts) > 3 else 10
        count = 0
        for title, nbook in numeric_torah_cache.items():
            for ch, vs, total in find_verses_with_value(nbook, target):
                print(f"  {title} {ch}:{vs} = {total}")
                count += 1
                if count >= limit:
                    break
            if count >= limit:
                break
        print(f"\n({count} match(es) shown, limit={limit})")
        return numeric_torah_cache

    # skip-sum
    if len(parts) < 3:
        print("Usage: /gematria skip-sum <target> [length] [min_skip] [max_skip] [limit]")
        return numeric_torah_cache
    target = int(parts[2])
    length = int(parts[3]) if len(parts) > 3 else 3
    min_skip = int(parts[4]) if len(parts) > 4 else -50
    max_skip = int(parts[5]) if len(parts) > 5 else 50
    limit = int(parts[6]) if len(parts) > 6 else 10

    count = 0
    for title, nbook in numeric_torah_cache.items():
        for m in find_skip_sum(nbook, target, length, min_skip, max_skip):
            print(f"  {m.book} skip={m.skip:+d} len={m.length}  "
                  f"{m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]} = {m.total}")
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    print(f"\n({count} match(es) shown, limit={limit})")
    return numeric_torah_cache


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
    print("Type '/elscode <hebrew word> [min_skip] [max_skip]' for ELS ('Bible code') search.")
    print("Type '/gematria word|verse-sum|skip-sum ...' for gematria numeric search.")
    print()

    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've run: python3 merge_model.py")
        sys.exit(1)

    print("\nReady! Ask your question below.\n")

    torah_cache = None
    numeric_torah_cache = None

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "סיום"]:
                print("\nShalom! שלום עד מפגש נוסף!")
                break

            if user_input.startswith("/elscode"):
                torah_cache = handle_els_command(user_input, model, tokenizer, torah_cache)
                continue

            if user_input.startswith("/gematria"):
                numeric_torah_cache = handle_gematria_command(user_input, numeric_torah_cache)
                continue

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