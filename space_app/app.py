import gradio as gr
import json
import os
import numpy as np
import torch
import chromadb
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(__file__))
from prompts import build_messages
from retrieval import retrieve
from els_search import load_torah, search_all_books, render_matrix
from gematria import find_skip_sum, find_verses_with_value, load_numeric_torah, word_value

MODEL_ID = "tdw419/judaism-llm-qwen2.5-7b"
DATASET_ID = "tdw419/judaism-llm-rag-corpus"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

# Globals (lazy-loaded on startup)
embedding_model = None
model = None
tokenizer = None
collection = None
_torah_cache = None
_numeric_torah_cache = None


def _get_torah():
    global _torah_cache
    if _torah_cache is None:
        _torah_cache = load_torah()
    return _torah_cache


def _get_numeric_torah():
    global _numeric_torah_cache
    if _numeric_torah_cache is None:
        _numeric_torah_cache = load_numeric_torah()
    return _numeric_torah_cache


def load_all():
    global embedding_model, model, tokenizer, collection
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Downloading RAG corpus from HF dataset...")
    seg_path = hf_hub_download(DATASET_ID, "segments.jsonl", repo_type="dataset")
    emb_path = hf_hub_download(DATASET_ID, "sefaria_embeddings.npy", repo_type="dataset")

    segments = [json.loads(l) for l in open(seg_path, encoding="utf-8")]
    embeddings = np.load(emb_path)
    print(f"Loaded {len(segments)} segments, embeddings {embeddings.shape}")

    # In-memory ChromaDB from the downloaded artifacts
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("sefaria_texts")
    if collection.count() == 0:
        B = 2000
        for i in range(0, len(segments), B):
            batch = segments[i:i + B]
            collection.add(
                embeddings=embeddings[i:i + B].tolist(),
                documents=[s["text"] for s in batch],
                metadatas=[{"source": s["source"], "category": s["category"],
                            "language": s["language"], "chunk_id": s["chunk_id"]} for s in batch],
                ids=[f"seg_{i + j}" for j in range(len(batch))],
            )
    print(f"Indexed {collection.count()} segments")

    print(f"Loading generation model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print("All models loaded")


def respond(query, history):
    if collection is None:
        return "Still loading models, try again in a minute..."
    r = retrieve(query, collection, embedding_model, top_k=5)
    docs = r["documents"][0] if r.get("documents") else []
    metas = r["metadatas"][0] if r.get("metadatas") else []
    messages = build_messages(query, docs, metas)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    src = "\n\n".join(f"[{i+1}] {m['source']}" for i, m in enumerate(metas))
    return f"{response}\n\n---\nSources:\n{src}"


def els_search_ui(word, min_skip, max_skip):
    word = (word or "").strip()
    if not word:
        return "Enter a Hebrew word to search for."
    try:
        min_skip, max_skip = int(min_skip), int(max_skip)
    except ValueError:
        return "Skip range must be integers."

    torah = _get_torah()
    matches = search_all_books(torah, word, min_skip, max_skip)
    lines = [ELS_DISCLAIMER, "",
             f"Found {len(matches)} match(es) for '{word}' across skips [{min_skip}, {max_skip}]", ""]
    for m in matches[:10]:
        lines.append(f"  {m.book} skip={m.skip:+d}  {m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]}")

    if not matches:
        return "\n".join(lines)

    top = matches[0]
    lines.append("")
    lines.append(render_matrix(torah[top.book], top))

    if collection is not None:
        query = f"{top.book} {top.start_ref[0]}:{top.start_ref[1]} {word}"
        r = retrieve(query, collection, embedding_model, top_k=3)
        docs = r["documents"][0] if r.get("documents") else []
        metas = r["metadatas"][0] if r.get("metadatas") else []
        if docs:
            lines.append("")
            lines.append(f"Related commentary near {top.book} {top.start_ref[0]}:{top.start_ref[1]}:")
            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                src = (meta or {}).get("source", "?")
                snippet = (doc or "").strip().replace("\n", " ")[:200]
                lines.append(f"  [{i}] ({src}) {snippet}...")

    return "\n".join(lines)


def gematria_word_ui(word):
    word = (word or "").strip()
    if not word:
        return "Enter a Hebrew word."
    return f"{GEMATRIA_DISCLAIMER}\n\n{word} = {word_value(word)}"


def gematria_verse_sum_ui(target, limit):
    try:
        target, limit = int(target), int(limit)
    except ValueError:
        return "Target and limit must be integers."

    torah = _get_numeric_torah()
    lines = [GEMATRIA_DISCLAIMER, ""]
    count = 0
    for title, nbook in torah.items():
        for ch, vs, total in find_verses_with_value(nbook, target):
            lines.append(f"  {title} {ch}:{vs} = {total}")
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    lines.append(f"\n({count} match(es) shown, limit={limit})")
    return "\n".join(lines)


def gematria_skip_sum_ui(target, length, min_skip, max_skip, limit):
    try:
        target, length, min_skip, max_skip, limit = (
            int(target), int(length), int(min_skip), int(max_skip), int(limit)
        )
    except ValueError:
        return "All fields must be integers."

    torah = _get_numeric_torah()
    lines = [GEMATRIA_DISCLAIMER, ""]
    count = 0
    for title, nbook in torah.items():
        for m in find_skip_sum(nbook, target, length, min_skip, max_skip):
            lines.append(f"  {m.book} skip={m.skip:+d} len={m.length}  "
                          f"{m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]} = {m.total}")
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    lines.append(f"\n({count} match(es) shown, limit={limit})")
    return "\n".join(lines)


if os.environ.get("HF_SPACE") == "1" or True:  # always preload in Space
    load_all()

with gr.Blocks(title="Judaism LLM — Sefaria RAG") as demo:
    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=respond,
            title="Judaism LLM — Sefaria RAG",
            description="Ask about Jewish texts. Answers are verbatim quotes from retrieved Sefaria passages with [N] citations, or an honest refusal if nothing retrieved is relevant.",
            examples=["What is Teshuva?", "מה המשמעות של יום כיפור", "Explain Shabbat", "תורה"],
        )

    with gr.Tab("ELS Search"):
        gr.Markdown(
            "### Equidistant Letter Sequence ('Bible code') search\n"
            + ELS_DISCLAIMER
        )
        els_word = gr.Textbox(label="Hebrew word", placeholder="תורה")
        with gr.Row():
            els_min = gr.Number(label="Min skip", value=-100, precision=0)
            els_max = gr.Number(label="Max skip", value=100, precision=0)
        els_button = gr.Button("Search")
        els_output = gr.Textbox(label="Results", lines=20)
        els_button.click(els_search_ui, [els_word, els_min, els_max], els_output)

    with gr.Tab("Gematria"):
        gr.Markdown("### Traditional gematria (mispar hechrachi) numeric search\n" + GEMATRIA_DISCLAIMER)
        with gr.Tab("Word value"):
            gem_word = gr.Textbox(label="Hebrew word", placeholder="תורה")
            gem_word_button = gr.Button("Compute")
            gem_word_output = gr.Textbox(label="Result")
            gem_word_button.click(gematria_word_ui, [gem_word], gem_word_output)

        with gr.Tab("Verse sum search"):
            gem_vs_target = gr.Number(label="Target value", value=611, precision=0)
            gem_vs_limit = gr.Number(label="Max results", value=10, precision=0)
            gem_vs_button = gr.Button("Search")
            gem_vs_output = gr.Textbox(label="Results", lines=15)
            gem_vs_button.click(gematria_verse_sum_ui, [gem_vs_target, gem_vs_limit], gem_vs_output)

        with gr.Tab("Skip sum search"):
            gem_ss_target = gr.Number(label="Target sum", value=26, precision=0)
            with gr.Row():
                gem_ss_length = gr.Number(label="Run length (letters)", value=3, precision=0)
                gem_ss_min = gr.Number(label="Min skip", value=-50, precision=0)
                gem_ss_max = gr.Number(label="Max skip", value=50, precision=0)
            gem_ss_limit = gr.Number(label="Max results", value=10, precision=0)
            gem_ss_button = gr.Button("Search")
            gem_ss_output = gr.Textbox(label="Results", lines=15)
            gem_ss_button.click(
                gematria_skip_sum_ui,
                [gem_ss_target, gem_ss_length, gem_ss_min, gem_ss_max, gem_ss_limit],
                gem_ss_output,
            )

if __name__ == "__main__":
    demo.launch()
