#!/usr/bin/env python3
"""
Verify that ChromaDB now stores actual text instead of chunk_id
"""

import chromadb
import numpy as np

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("sefaria_texts")

print(f"Collection count: {collection.count()}")

# Test retrieval with dummy embedding (just check document field has text)
test_query = np.random.randn(384).tolist()

results = collection.query(
    query_embeddings=[test_query],
    n_results=3
)

print(f"\nRetrieved {len(results['ids'][0])} results:\n")

for i in range(len(results['ids'][0])):
    doc_id = results['ids'][0][i]
    document = results['documents'][0][i]
    metadata = results['metadatas'][0][i]

    print(f"Result {i+1}:")
    print(f"  ID: {doc_id}")
    print(f"  Source: {metadata['source']}")
    print(f"  Language: {metadata['language']}")

    # Check if document is actual text or just a chunk_id
    if len(document) < 50 and '_' in document:
        print(f"  ⚠ WARNING: Document is a chunk_id, not actual text: {document}")
    elif len(document) > 100:
        print(f"  ✓ Document is actual text (length: {len(document)} chars)")
        print(f"  First 200 chars: {document[:200]}...")
    else:
        print(f"  ? Document length: {len(document)} chars")
        print(f"  Content: {document[:100]}...")
    print()

# Summary check
print("="*60)
has_text = 0
has_chunk_id = 0

for doc in results['documents'][0]:
    if len(doc) < 50 and '_' in doc:
        has_chunk_id += 1
    elif len(doc) > 100:
        has_text += 1

print(f"Summary:")
print(f"  Results with actual text: {has_text}")
print(f"  Results with chunk_id only: {has_chunk_id}")

if has_text > has_chunk_id:
    print("\n✓ PASS: ChromaDB is storing actual text")
elif has_chunk_id > 0:
    print("\n✗ FAIL: ChromaDB is still storing chunk_id")
else:
    print("\n? UNCLEAR: Cannot determine document content")