#!/usr/bin/env python3
"""
Phase 2: Vector Database Setup
Set up ChromaDB and index Sefaria embeddings
"""

import json
import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path
import sys

# Configuration
EMBEDDINGS_FILE = "sefaria_embeddings.npy"
METADATA_FILE = "sefaria_metadata.jsonl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "sefaria_texts"

def main():
    print("Phase 2: Vector Database Setup (ChromaDB)")

    # Check files exist
    if not Path(EMBEDDINGS_FILE).exists():
        print(f"Error: {EMBEDDINGS_FILE} not found. Run Phase 1 first.")
        return False

    if not Path(METADATA_FILE).exists():
        print(f"Error: {METADATA_FILE} not found. Run Phase 1 first.")
        return False

    # Load data
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded {embeddings.shape[0]} embeddings")

    print(f"Loading metadata from {METADATA_FILE}...")
    metadata = []
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line))
    print(f"Loaded {len(metadata)} metadata entries")

    print(f"Loading segments text from segments.jsonl...")
    segments = []
    with open("segments.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            segments.append(json.loads(line))
    print(f"Loaded {len(segments)} segments with text")

    if len(embeddings) != len(metadata) or len(embeddings) != len(segments):
        print(f"Error: Mismatch between embeddings ({len(embeddings)}), metadata ({len(metadata)}), and segments ({len(segments)})")
        return False

    # Setup ChromaDB
    print(f"\nSetting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection")
    except:
        pass

    # Create collection
    print(f"Creating collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Sefaria texts for Judaism LLM RAG"}
    )

    # Add embeddings in batches
    print(f"\nIndexing {len(embeddings)} segments...")
    batch_size = 1000

    for i in range(0, len(embeddings), batch_size):
        batch_end = min(i + batch_size, len(embeddings))

        batch_embeddings = embeddings[i:batch_end].tolist()
        batch_ids = [f"seg_{j}" for j in range(i, batch_end)]
        batch_metadata = [metadata[j] for j in range(i, batch_end)]
        batch_documents = [segments[j]['text'] for j in range(i, batch_end)]

        collection.add(
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metadata,
            documents=batch_documents
        )

        print(f"  Indexed {batch_end}/{len(embeddings)} segments")

    # Verify collection
    print(f"\nVerifying collection...")
    count = collection.count()
    print(f"Total documents in collection: {count}")

    # Test retrieval
    print(f"\nTesting retrieval...")
    test_query = np.random.randn(384).tolist()  # Dummy query

    results = collection.query(
        query_embeddings=[test_query],
        n_results=3
    )

    print(f"Retrieved {len(results['ids'][0])} results")
    print(f"Sample result:")
    print(f"  ID: {results['ids'][0][0]}")
    print(f"  Source: {results['metadatas'][0][0]['source']}")
    print(f"  Language: {results['metadatas'][0][0]['language']}")

    print(f"\n✓ Phase 2 Complete:")
    print(f"  ChromaDB directory: {CHROMA_DIR}/")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Indexed: {count} segments")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)