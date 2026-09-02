#!/usr/bin/env python3
"""
Improved RAG evaluation with grounding metrics
Measures whether responses are grounded in retrieved context
"""

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import numpy as np
from pathlib import Path
import sys
import re

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sefaria_texts"
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EVALUATION_REPORT = "RAG_EVALUATION_IMPROVED.md"

# Test queries (expanded to 10)
TEST_QUERIES = [
    {"query": "What is Teshuva?", "expected_concepts": ["repentance", "return", "sin", "atonement"], "language": "english"},
    {"query": "Explain Shabbat", "expected_concepts": ["rest", "seventh day", "sanctified", "work"], "language": "english"},
    {"query": "What are the main sources of Jewish law?", "expected_concepts": ["Torah", "Oral Law", "Mishnah", "Talmud"], "language": "english"},
    {"query": "תורה", "expected_concepts": ["תורה", "חומש", "חמשה חומשי תורה"], "language": "hebrew"},
    {"query": "משנה", "expected_concepts": ["משנה", "תורה שבעל פה", "תנאים"], "language": "hebrew"},
    {"query": "What is the significance of Rosh Hashanah?", "expected_concepts": ["new year", "judgment", "shofar", "creation"], "language": "english"},
    {"query": "Explain the concept of Kashrut", "expected_concepts": ["kosher", "dietary laws", "permitted", "forbidden"], "language": "english"},
    {"query": "מה המשמעות של יום כיפור", "expected_concepts": ["יום כיפור", "כפרה", "צום", "סליחה"], "language": "hebrew"},
    {"query": "פסח", "expected_concepts": ["פסח", "יציאת מצרים", "מצה", "חג"], "language": "hebrew"},
    {"query": "What are the Ten Commandments?", "expected_concepts": ["commandments", "sinai", "moses", "decalogue"], "language": "english"}
]

def calculate_ngram_overlap(response, context, n=3):
    """Calculate n-gram overlap between response and retrieved context."""
    # Normalize text
    response_words = re.findall(r'\w+', response.lower())
    context_words = re.findall(r'\w+', context.lower())

    # Generate n-grams
    response_ngrams = set(' '.join(response_words[i:i+n]) for i in range(len(response_words)-n+1))
    context_ngrams = set(' '.join(context_words[i:i+n]) for i in range(len(context_words)-n+1))

    if not response_ngrams:
        return 0.0

    overlap = len(response_ngrams & context_ngrams)
    coverage = overlap / len(response_ngrams)

    return coverage

def check_grounding(results):
    """Check if response quotes from retrieved context."""
    total_results = 0
    grounded_results = 0

    for r in results:
        if r["retrieved_docs"] and r["response"]:
            context = " ".join(r["retrieved_docs"])
            overlap_3 = calculate_ngram_overlap(r["response"], context, n=3)
            overlap_5 = calculate_ngram_overlap(r["response"], context, n=5)

            # Consider grounded if >20% 5-gram overlap or >30% 3-gram overlap
            is_grounded = (overlap_5 > 0.2) or (overlap_3 > 0.3)

            r["grounding_3gram"] = overlap_3
            r["grounding_5gram"] = overlap_5
            r["is_grounded"] = is_grounded

            total_results += 1
            if is_grounded:
                grounded_results += 1

    return grounded_results / total_results if total_results else 0

def check_concept_coverage(results, test_queries):
    """Check if responses contain expected concepts."""
    total_concepts = 0
    covered_concepts = 0

    for i, r in enumerate(results):
        if not r["response"]:
            continue

        expected = test_queries[i]["expected_concepts"]
        total_concepts += len(expected)

        for concept in expected:
            if concept.lower() in r["response"].lower():
                covered_concepts += 1

    return covered_concepts / total_concepts if total_concepts else 0

def generate_improved_report(results, grounding_score, concept_coverage):
    """Generate improved evaluation report."""
    report = f"""# RAG Evaluation Report (Improved Metrics)

## Overview

- **Total Queries:** {len(results)}
- **Grounding Score:** {grounding_score:.1%} (response n-grams in retrieved context)
- **Concept Coverage:** {concept_coverage:.1%} (expected concepts in responses)
- **Average Response Length:** {np.mean([len(r['response']) for r in results if r['response']]):.0f} characters

## Test Results

| Query | Response Length | Grounding (5-gram) | Grounding (3-gram) | Is Grounded? | Concepts Covered |
|-------|----------------|-------------------|-------------------|--------------|------------------|
"""

    for i, r in enumerate(results, 1):
        query = TEST_QUERIES[i-1]["query"]
        response_length = len(r["response"])
        grounding_5 = r.get("grounding_5gram", 0)
        grounding_3 = r.get("grounding_3gram", 0)
        is_grounded = r.get("is_grounded", False)

        expected = TEST_QUERIES[i-1]["expected_concepts"]
        covered = sum(1 for c in expected if c.lower() in r["response"].lower())
        total = len(expected)

        report += f"| {query} | {response_length} | {grounding_5:.1%} | {grounding_3:.1%} | {'✓' if is_grounded else '✗'} | {covered}/{total} |\n"

    report += f"""
## Metric Definitions

### Grounding Score
- Measures n-gram overlap between response and retrieved context
- 5-gram overlap >20% OR 3-gram overlap >30% = grounded
- Higher score = more response comes from retrieved text
- Addresses hallucination issue

### Concept Coverage
- Measures if response contains expected key concepts
- Expected concepts defined per query
- Higher score = more complete information

## Conclusions

"""

    if grounding_score > 0.7:
        report += "✓ RAG system is well-grounded (grounding > 70%)\n"
    elif grounding_score > 0.4:
        report += "⚠ RAG system partially grounded (grounding 40-70%)\n"
    else:
        report += "✗ RAG system poorly grounded (grounding < 40%)\n"

    if concept_coverage > 0.7:
        report += "✓ Concept coverage is high (coverage > 70%)\n"
    elif concept_coverage > 0.4:
        report += "⚠ Concept coverage is moderate (coverage 40-70%)\n"
    else:
        report += "✗ Concept coverage is low (coverage < 40%)\n"

    report += f"""
## Recommendations

1. **If grounding score is low:**
   - Increase top_k in retrieval
   - Improve embedding model quality
   - Check if retrieved context is actually relevant

2. **If concept coverage is low:**
   - Increase context window (more retrieved text)
   - Improve generation prompt
   - Check if model has sufficient knowledge

3. **If both are low:**
   - System may not be retrieving relevant passages
   - Consider expanding corpus or improving embeddings

4. **If grounding is high but coverage is low:**
   - Model may be too conservative (sticking too close to text)
   - Encourage more synthesis with prompt engineering

---

**Generated:** September 1, 2026
**Model:** judaism-llm-qwen2.5-7b-merged
**Corpus:** 18,453 Sefaria segments
**Improvement:** Added grounding metrics to replace source_accuracy
"""

    return report

def main():
    """Run improved evaluation suite."""
    print("Phase 6: Improved Evaluation Suite\n")

    # Load models (CPU only for evaluation to avoid OOM)
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading generation model...")
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

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]

        print(f"Query {i+1}: {query}")
        print("-" * 60)

        # Embed query
        query_embedding = embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        # Vector search
        r_results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)

        if not r_results['ids'][0]:
            print("Error: No relevant passages found")
            results.append({
                "query": query,
                "response": "",
                "sources": [],
                "retrieved_docs": []
            })
            continue

        # Build context from retrieved documents
        context_parts = []
        sources = []

        for j in range(len(r_results['ids'][0])):
            metadata = r_results['metadatas'][0][j]
            document = r_results['documents'][0][j]

            context_parts.append(document)
            sources.append(metadata['source'])

        context = "\n---\n".join(context_parts)

        # Generate response
        messages = [
            {"role": "system", "content": "You are Judaism LLM. Answer based on texts with citations."},
            {"role": "user", "content": f"Texts:\n{context}\n\nQuestion: {query}\n\nAnswer with citations."}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, seed=42)

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        print(f"Response: {response[:200]}...")
        print(f"Sources: {sources}")
        print()

        results.append({
            "query": query,
            "response": response,
            "sources": sources,
            "retrieved_docs": context_parts
        })

    # Calculate improved metrics
    print("Calculating grounding metrics...")
    grounding_score = check_grounding(results)

    print("Calculating concept coverage...")
    concept_coverage = check_concept_coverage(results, TEST_QUERIES)

    print(f"\n=== Improved Metrics ===")
    print(f"Grounding Score: {grounding_score:.1%}")
    print(f"Concept Coverage: {concept_coverage:.1%}")
    print(f"Avg Response Length: {np.mean([len(r['response']) for r in results if r['response']]):.0f} chars")

    # Generate improved report
    print(f"\nGenerating improved report...")
    report = generate_improved_report(results, grounding_score, concept_coverage)

    with open(EVALUATION_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved to: {EVALUATION_REPORT}")
    print("\n✓ Phase 6 Complete (Improved):")
    print("  Grounding metrics implemented")
    print(f"  Grounding Score: {grounding_score:.1%}")
    print(f"  Concept Coverage: {concept_coverage:.1%}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)