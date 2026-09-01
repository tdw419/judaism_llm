#!/usr/bin/env python3
"""
Phase 1: Data Preparation - Memory-efficient segmentation and embedding
Processes Sefaria texts in batches to avoid OOM
"""

import json
from pathlib import Path
import gc
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

# Configuration
SEFARIA_DIR = "sefaria_texts"
SEGMENTS_FILE = "segments.jsonl"
EMBEDDINGS_FILE = "sefaria_embeddings.npy"
METADATA_FILE = "sefaria_metadata.jsonl"
BATCH_SIZE = 100  # Process files in batches
MAX_TOKENS = 512
OVERLAP_TOKENS = 50

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def segment_text(text: str, source: str, category: str, language: str, idx: int):
    """Segment text into chunks (simplified, no overlap for memory)."""
    if estimate_tokens(text) <= MAX_TOKENS:
        return [{
            "text": text,
            "source": source,
            "category": category,
            "language": language,
            "chunk_id": f"{source}_{idx}_0"
        }]

    # Simple chunking at character boundaries
    chunk_size = MAX_TOKENS * 4
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i+chunk_size].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "category": category,
                "language": language,
                "chunk_id": f"{source}_{idx}_{len(chunks)}"
            })
    return chunks

def process_batch(files):
    """Process a batch of files."""
    segments = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            continue

        source = file_path.stem
        category = file_path.parent.name

        he_text = ' '.join([t for t in data.get('he', []) if isinstance(t, str)])
        en_text = ' '.join([t for t in data.get('text', []) if isinstance(t, str)])

        if he_text and estimate_tokens(he_text) > 10:
            segments.extend(segment_text(he_text, source, category, "hebrew", 0))
        if en_text and estimate_tokens(en_text) > 10:
            segments.extend(segment_text(en_text, source, category, "english", 1))

    return segments

def main():
    print("Phase 1: Data Preparation (Memory-Efficient)")

    sefaria_path = Path(SEFARIA_DIR)
    if not sefaria_path.exists():
        print(f"Error: {SEFARIA_DIR} not found")
        return False

    json_files = list(sefaria_path.rglob("*.json"))
    print(f"Found {len(json_files)} Sefaria JSON files")

    # Process in batches to avoid OOM
    all_segments = []
    for i in range(0, len(json_files), BATCH_SIZE):
        batch = json_files[i:i+BATCH_SIZE]
        segments = process_batch(batch)
        all_segments.extend(segments)
        print(f"Processed {i+len(batch)}/{len(json_files)} files ({len(all_segments)} segments)")
        gc.collect()  # Force garbage collection

    print(f"Total segments: {len(all_segments)}")

    # Save segments
    print(f"Saving segments to {SEGMENTS_FILE}...")
    with open(SEGMENTS_FILE, 'w', encoding='utf-8') as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + '\n')

    # Embedding (smaller batch size)
    print(f"\nLoading embedding model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Embedding {len(all_segments)} segments...")
    embeddings = []
    texts = [s['text'] for s in all_segments]

    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        batch_embeddings = model.encode(
            batch,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings.append(batch_embeddings)
        print(f"Embedded {i+len(batch)}/{len(texts)} segments")
        gc.collect()

    final_embeddings = np.vstack(embeddings)

    # Save embeddings
    print(f"\nSaving embeddings to {EMBEDDINGS_FILE}...")
    np.save(EMBEDDINGS_FILE, final_embeddings)

    # Save metadata
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(all_segments):
            metadata = {
                "index": i,
                "chunk_id": seg['chunk_id'],
                "source": seg['source'],
                "category": seg['category'],
                "language": seg['language']
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

    hebrew_count = sum(1 for s in all_segments if s['language'] == 'hebrew')
    english_count = sum(1 for s in all_segments if s['language'] == 'english')

    print(f"\n✓ Phase 1 Complete:")
    print(f"  Hebrew segments: {hebrew_count}")
    print(f"  English segments: {english_count}")
    print(f"  Total: {len(all_segments)}")
    print(f"  Embeddings shape: {final_embeddings.shape}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)