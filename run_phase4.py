#!/usr/bin/env python3
"""
Phase 4: Interactive CLI
Hebrew/English chat with real-time retrieval + generation
"""

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sefaria_texts"
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_models():
    """Load all required models."""
    print("Loading models (this may take a minute)...")

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

    return embedding_model, model, tokenizer

def detect_language(text):
    """Simple Hebrew/English detection."""
    hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    return "Hebrew" if hebrew_chars > len(text) * 0.3 else "English"

def rag_query(query, embedding_model, model, tokenizer, collection):
    """Execute RAG query with timing."""
    import time

    # Step 1: Embed query
    start_retrieve = time.time()
    query_embedding = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

    # Step 2: Vector search
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=TOP_K)

    if not results['ids'][0]:
        return {"error": "No relevant passages found", "sources": [], "time": 0}

    # Step 3: Context assembly
    context_parts = []
    sources = []

    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        source_info = f"[{metadata['source']}]"
        context_parts.append(f"{source_info} {document}")
        sources.append(f"{metadata['source']} ({metadata['language']})")

    context = "\n---\n".join(context_parts)

    # Step 4: Generation
    messages = [
        {
            "role": "system",
            "content": "You are Judaism LLM, trained on Sefaria corpus. Answer based on provided texts with citations."
        },
        {
            "role": "user",
            "content": f"Texts:\n{context}\n\nQuestion: {query}\n\nAnswer with citations."
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    retrieval_time = time.time() - start_retrieve

    return {"response": response, "sources": sources, "time": retrieval_time}

def main():
    """Interactive chat loop."""
    print("=" * 70)
    print("     Judaism LLM - Interactive RAG Chat")
    print("=" * 70)
    print("Ask questions about Jewish texts in English or Hebrew.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    # Load models
    embedding_model, model, tokenizer = load_models()

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✓ Connected (documents: {collection.count()})")
    except:
        print(f"✗ Error: Collection not found. Run Phase 2 first.")
        return False

    print("\nReady! Ask your question below.\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("\nShalom!")
                break

            lang = detect_language(user_input)
            print(f"\n[{lang}] Retrieving and generating...\n")

            result = rag_query(user_input, embedding_model, model, tokenizer, collection)

            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Response:\n{result['response']}\n")
                print(f"Sources: {', '.join(result['sources'][:5])}")
                print(f"Time: {result['time']:.2f}s")

            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nShalom!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)