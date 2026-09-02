#!/usr/bin/env python3
"""
Phase 5: Web Interface
Simple FastAPI backend with HTML/JS frontend
"""

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

from prompts import build_messages

# Configuration (env-overridable for Docker; see Dockerfile)
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "sefaria_texts"
MODEL_PATH = os.environ.get("MODEL_PATH", "judaism-llm-qwen2.5-7b-merged")
TOP_K = int(os.environ.get("TOP_K", "5"))
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Initialize FastAPI
app = FastAPI(title="Judaism LLM RAG")

# CORS (allow_credentials=True is incompatible with wildcard origins per the
# spec - browsers drop credentialed responses. This API doesn't use cookies,
# so credentials stay off and wildcard is valid.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global models (lazy loaded)
embedding_model = None
model = None
tokenizer = None
collection = None

def load_models():
    """Load models on startup."""
    global embedding_model, model, tokenizer, collection

    print("Loading models...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

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

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"✓ Models loaded (collection: {collection.count()} documents)")

@app.on_event("startup")
async def startup():
    """Load models on startup."""
    load_models()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve frontend."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Judaism LLM RAG</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        input, button { padding: 10px; margin: 5px; }
        .response { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .sources { margin-top: 10px; color: #666; font-size: 14px; }
        .loading { color: #666; }
        .error { color: #d32f2f; }
    </style>
</head>
<body>
    <h1>🕎 Judaism LLM RAG</h1>
    <p>Ask questions about Jewish texts (Torah, Talmud, etc.)</p>

    <input type="text" id="query" placeholder="Your question..." style="width: 80%">
    <button onclick="query()">Ask</button>

    <div id="result"></div>

    <script>
        async function query() {
            const query = document.getElementById('query').value;
            if (!query) return;

            const result = document.getElementById('result');
            result.innerHTML = '<div class="loading">Retrieving and generating...</div>';

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                });

                const data = await response.json();

                if (data.error) {
                    result.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                } else {
                    const sourcesHtml = data.sources.map(s => `<span>${s}</span>`).join(' | ');
                    result.innerHTML = `
                        <div class="response">
                            <strong>Response:</strong><br><br>
                            ${data.response.replace(/\n/g, '<br>')}
                            <br><br><div class="sources"><strong>Sources:</strong> ${sourcesHtml}</div>
                        </div>
                    `;
                }
            } catch (e) {
                result.innerHTML = `<div class="error">Error: ${e.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.post("/query")
async def query(request: dict):
    """RAG query endpoint."""
    query = request.get("query", "")
    top_k = request.get("top_k", TOP_K)

    if not query:
        return {"error": "Query required"}

    try:
        # Embed query
        query_embedding = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        # Vector search
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        if not results['ids'][0]:
            return {"error": "No relevant passages found"}

        # Source list for the response payload
        sources = []

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            sources.append(f"{metadata['source']} ({metadata['language']})")

        # Generation (extractive: verbatim-quote, cite, or refuse)
        messages = build_messages(query, results['documents'][0], results['metadatas'][0])

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        return {"response": response, "sources": sources}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)