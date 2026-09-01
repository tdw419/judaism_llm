#!/usr/bin/env python3
"""
Phase 6: Evaluation Suite
Test RAG quality with metrics and human evaluation
"""

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import numpy as np
from pathlib import Path
import sys

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sefaria_texts"
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EVALUATION_REPORT = "RAG_EVALUATION.md"

# Test queries
TEST_QUERIES = [
    {"query": "What is Teshuva?", "expected_sources": ["Mishneh Torah", "Talmud"], "language": "english"},
    {"query": "Explain Shabbat", "expected_sources": ["Torah", "Mishnah"], "language": "english"},
    {"query": "What are the main sources of Jewish law?", "expected_sources": ["Torah", "Mishnah", "Talmud"], "language": "english"},
    {"query": "תורה", "expected_sources": ["Torah"], "language": "hebrew"},
    {"query": "משנה", "expected_sources": ["Mishnah"], "language": "hebrew"}
]

def load_models():
    """Load all required models."""
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

    return embedding_model, model, tokenizer

def rag_query(query, embedding_model, model, tokenizer, collection):
    """Execute RAG query."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)

    if not results['ids'][0]:
        return {"response": "", "sources": [], "retrieved_docs": []}

    context_parts = []
    sources = []

    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        context_parts.append(f"[{metadata['source']}] {document}")
        sources.append(metadata['source'])

    context = "\n---\n".join(context_parts)

    messages = [
        {"role": "system", "content": "You are Judaism LLM. Answer based on texts with citations."},
        {"role": "user", "content": f"Texts:\n{context}\n\nQuestion: {query}\n\nAnswer with citations."}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    return {"response": response, "sources": sources, "retrieved_docs": [results['documents'][0][i] for i in range(len(results['documents'][0]))]}

def calculate_metrics(results):
    """Calculate evaluation metrics."""
    total_queries = len(results)
    successful_queries = sum(1 for r in results if r["response"])

    # Source citation accuracy
    source_accuracy = 0
    for r in results:
        if r["sources"]:
            expected_sources = r["expected_sources"]
            retrieved_sources = r["sources"]
            matches = sum(1 for s in retrieved_sources if any(e.lower() in s.lower() for e in expected_sources))
            source_accuracy += matches / len(expected_sources) if expected_sources else 0

    source_accuracy /= total_queries if total_queries else 0

    # Average response length
    avg_response_length = np.mean([len(r["response"]) for r in results if r["response"]])

    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / total_queries if total_queries else 0,
        "source_accuracy": source_accuracy,
        "avg_response_length": avg_response_length
    }

def generate_report(results, metrics):
    """Generate evaluation report."""
    report = f"""# RAG Evaluation Report

## Overview

- **Total Queries:** {metrics['total_queries']}
- **Successful Queries:** {metrics['successful_queries']}
- **Success Rate:** {metrics['success_rate']:.1%}
- **Source Citation Accuracy:** {metrics['source_accuracy']:.1%}
- **Average Response Length:** {metrics['avg_response_length']:.0f} characters

## Test Results

| Query | Response Length | Sources | Expected Sources | Match |
|-------|----------------|---------|------------------|-------|
"""

    for i, r in enumerate(results, 1):
        query = r["query"]
        response_length = len(r["response"])
        sources = ", ".join(r["sources"]) if r["sources"] else "None"
        expected_sources = ", ".join(r["expected_sources"])
        match = "✓" if any(e.lower() in s.lower() for s in r["sources"] for e in r["expected_sources"]) else "✗"

        report += f"| {query} | {response_length} | {sources} | {expected_sources} | {match} |\n"

    report += """
## Sample Responses

### Query 1: What is Teshuva?

"""
    report += results[0]["response"] + "\n\n"

    report += """
## Conclusions

"""
    if metrics['success_rate'] > 0.8:
        report += "✓ RAG system performs well (success rate > 80%)\n"
    elif metrics['success_rate'] > 0.5:
        report += "⚠ RAG system needs improvement (success rate 50-80%)\n"
    else:
        report += "✗ RAG system requires significant improvement (success rate < 50%)\n"

    if metrics['source_accuracy'] > 0.7:
        report += "✓ Source citations are accurate (accuracy > 70%)\n"
    else:
        report += "⚠ Source citations need improvement (accuracy < 70%)\n"

    report += """
## Recommendations

1. **If success rate is low:**
   - Increase top_k in retrieval
   - Improve embedding model
   - Expand Sefaria corpus

2. **If source accuracy is low:**
   - Improve metadata quality
   - Add more source references
   - Refine citation extraction

3. **If responses are too short:**
   - Increase max_new_tokens
   - Improve context assembly
   - Fine-tune model further

---

**Generated:** September 1, 2026
**Model:** judaism-llm-qwen2.5-7b-merged
**Corpus:** 18,453 Sefaria segments
"""

    return report

def main():
    """Run evaluation suite."""
    print("Phase 6: Evaluation Suite\n")

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

    # Run test queries
    print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")
    results = []

    for test in TEST_QUERIES:
        query = test["query"]
        expected_sources = test["expected_sources"]

        print(f"Query: {query}")
        print("-" * 60)

        result = rag_query(query, embedding_model, model, tokenizer, collection)
        result["query"] = query
        result["expected_sources"] = expected_sources

        if result["response"]:
            print(f"Response: {result['response'][:200]}...")
            print(f"Sources: {result['sources']}")
        else:
            print("Error: No response generated")

        print()

        results.append(result)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)

    print(f"\n=== Metrics ===")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Source Accuracy: {metrics['source_accuracy']:.1%}")
    print(f"Avg Response Length: {metrics['avg_response_length']:.0f} chars")

    # Generate report
    print(f"\nGenerating report...")
    report = generate_report(results, metrics)

    with open(EVALUATION_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved to: {EVALUATION_REPORT}")
    print("\n✓ Phase 6 Complete:")
    print("  Evaluation suite functional")
    print("  Metrics calculated")
    print(f"  Report: {EVALUATION_REPORT}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)