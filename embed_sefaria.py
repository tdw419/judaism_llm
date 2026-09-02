#!/usr/bin/env python3
"""
Embed Sefaria segments for RAG retrieval using multilingual sentence-transformers
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# Configuration
SEGMENTS_FILE = "segments.jsonl"
EMBEDDINGS_FILE = "sefaria_embeddings.npy"
METADATA_FILE = "sefaria_metadata.jsonl"
BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_segments():
    """Load segments from JSONL file."""
    segments = []
    with open(SEGMENTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            segments.append(json.loads(line))
    return segments

def embed_segments(segments):
    """Embed segments using multilingual model."""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(segments)} segments...")
    texts = [s['text'] for s in segments]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings

def save_embeddings(embeddings, segments):
    """Save embeddings and metadata."""
    import numpy as np

    print(f"\nSaving embeddings to {EMBEDDINGS_FILE}...")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            metadata = {
                "index": i,
                "chunk_id": seg['chunk_id'],
                "source": seg['source'],
                "category": seg['category'],
                "language": seg['language']
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

    print(f"\n=== Summary ===")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Dimensions: {embeddings.shape[1]}")
    print(f"Segments: {len(segments)}")
    print(f"\nSaved:")
    print(f"  {EMBEDDINGS_FILE}")
    print(f"  {METADATA_FILE}")

def test_embeddings(embeddings, segments):
    """Test embedding quality with sample queries."""
    from sentence_transformers import util

    print("\n=== Testing Embeddings ===")

    test_queries = [
        "What is Teshuva?",
        "Explain Shabbat",
        "תורה",
        "משנה"
    ]

    model = SentenceTransformer(MODEL_NAME)

    for query in test_queries:
        query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        # Find top 3 similar segments
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)

        print(f"\nQuery: {query}")
        for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
            segment = segments[idx.item()]
            print(f"  {i+1}. Score: {score:.4f} | {segment['language']} | {segment['source']}")
            print(f"     Text: {segment['text'][:100]}...")

def main():
    """Main embedding pipeline."""
    print("=== Sefaria Embedding Pipeline ===\n")

    # Load segments
    print(f"Loading segments from {SEGMENTS_FILE}...")
    segments = load_segments()
    print(f"Loaded {len(segments)} segments\n")

    # Embed
    embeddings = embed_segments(segments)

    # Save
    save_embeddings(embeddings, segments)

    # Test
    test_embeddings(embeddings, segments)

    print("\n✓ Phase 1 complete: Segments embedded and ready for ChromaDB")

if __name__ == "__main__":
    main()