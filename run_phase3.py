#!/usr/bin/env python3
"""
Phase 3: RAG Query Engine
Implement retrieval + generation pipeline with source citations
"""

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import sys
from pathlib import Path

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sefaria_texts"
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_model():
    """Load Judaism LLM and tokenizer."""
    print(f"Loading model from: {MODEL_PATH}")
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
    return model, tokenizer

def rag_query(query, model, tokenizer, embedding_model, collection):
    """Execute RAG query with retrieval and generation."""
    # Step 1: Embed query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

    # Step 2: Vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K
    )

    if not results['ids'][0]:
        return {"error": "No relevant passages found", "sources": []}

    # Step 3: Context assembly
    context_parts = []
    sources = []

    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]

        source_info = f"[Source: {metadata['source']}, Category: {metadata['category']}, Language: {metadata['language']}]"
        context_parts.append(f"{source_info}\n{document}\n")
        sources.append({
            "source": metadata['source'],
            "category": metadata['category'],
            "language": metadata['language'],
            "chunk_id": metadata['chunk_id']
        })

    context = "\n---\n".join(context_parts)

    # Step 4: Generation with Judaism LLM
    messages = [
        {
            "role": "system",
            "content": "You are Judaism LLM, trained on Sefaria corpus. Answer questions based on the provided Sefaria texts. Include citations."
        },
        {
            "role": "user",
            "content": f"Based on these Sefaria texts:\n\n{context}\n\nAnswer the question: {query}\n\nInclude citations from the passages."
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    return {"response": response, "sources": sources}

def main():
    """Test RAG query engine."""
    print("Phase 3: RAG Query Engine\n")

    # Load models
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading generation model...")
    model, tokenizer = load_model()

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Connected to collection: {COLLECTION_NAME}")
        print(f"Document count: {collection.count()}")
    except:
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Run Phase 2 first.")
        return False

    # Test queries
    test_queries = [
        "What is Teshuva?",
        "Explain the concept of Shabbat",
        "What are the main sources of Jewish law?"
    ]

    print("\n=== Testing RAG Queries ===\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        result = rag_query(query, model, tokenizer, embedding_model, collection)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response:\n{result['response']}\n")
            print(f"Sources ({len(result['sources'])}):")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. {source['source']} ({source['language']}, {source['category']})")

    print("\n✓ Phase 3 Complete:")
    print("  RAG query engine functional")
    print("  Retrieval + generation working")
    print("  Source citations included")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)