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

MODEL_ID = "tdw419/judaism-llm-qwen2.5-7b"
DATASET_ID = "tdw419/judaism-llm-rag-corpus"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Globals (lazy-loaded on startup)
embedding_model = None
model = None
tokenizer = None
collection = None


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
    docs = r["documents"]
    metas = r["metadatas"]
    messages = build_messages(query, docs, metas)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    response = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    src = "\n\n".join(f"[{i+1}] {m['source']}" for i, m in enumerate(metas))
    return f"{response}\n\n---\nSources:\n{src}"


if os.environ.get("HF_SPACE") == "1" or True:  # always preload in Space
    load_all()

demo = gr.ChatInterface(
    fn=respond,
    title="Judaism LLM — Sefaria RAG",
    description="Ask about Jewish texts. Answers are verbatim quotes from retrieved Sefaria passages with [N] citations, or an honest refusal if nothing retrieved is relevant.",
    examples=["What is Teshuva?", "מה המשמעות של יום כיפור", "Explain Shabbat", "תורה"],
)

if __name__ == "__main__":
    demo.launch()
