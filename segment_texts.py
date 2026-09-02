#!/usr/bin/env python3
"""
Segment Sefaria texts for RAG retrieval
Split long texts into 512-token chunks with overlap
Preserve metadata: source, category, language
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Configuration
SEFARIA_DIR = "sefaria_texts"
OUTPUT_FILE = "segments.jsonl"
MAX_TOKENS = 512
OVERLAP_TOKENS = 50

# Simple token estimation (heuristic: ~4 chars per token for Hebrew/English)
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def segment_text(text: str, source: str, category: str, language: str) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    chunks = []
    tokens = estimate_tokens(text)

    if tokens <= MAX_TOKENS:
        chunks.append({
            "text": text,
            "source": source,
            "category": category,
            "language": language,
            "chunk_id": f"{source}_chunk_0"
        })
        return chunks

    # Split into chunks
    start_idx = 0
    chunk_idx = 0
    total_chars = len(text)
    chunk_chars = MAX_TOKENS * 4
    overlap_chars = OVERLAP_TOKENS * 4

    while start_idx < total_chars:
        end_idx = min(start_idx + chunk_chars, total_chars)

        # Try to break at sentence boundary
        if end_idx < total_chars:
            # Look for sentence endings: ., ?, !, and following space
            for delimiter in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
                last_delim = text.rfind(delimiter, start_idx, end_idx)
                if last_delim != -1:
                    end_idx = last_delim + 2
                    break

        chunk_text = text[start_idx:end_idx].strip()

        if chunk_text:
            chunk_id = hashlib.md5(f"{source}_{chunk_idx}".encode()).hexdigest()[:8]
            chunks.append({
                "text": chunk_text,
                "source": source,
                "category": category,
                "language": language,
                "chunk_id": f"{source}_{chunk_idx}"
            })
            chunk_idx += 1

        start_idx = end_idx - overlap_chars

    return chunks

def process_sefaria_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a single Sefaria JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    chunks = []

    # Extract source and category from file path
    parts = file_path.relative_to(SEFARIA_DIR).parts
    source = file_path.stem
    category = parts[0] if len(parts) > 0 else "Unknown"

    # Extract Hebrew text
    he_text = ""
    if 'he' in data and isinstance(data['he'], list):
        he_text = ' '.join([t for t in data['he'] if isinstance(t, str)])

    # Extract English text
    en_text = ""
    if 'text' in data and isinstance(data['text'], list):
        en_text = ' '.join([t for t in data['text'] if isinstance(t, str)])

    # Segment Hebrew
    if he_text and estimate_tokens(he_text) > 10:
        chunks.extend(segment_text(he_text, source, category, "hebrew"))

    # Segment English
    if en_text and estimate_tokens(en_text) > 10:
        chunks.extend(segment_text(en_text, source, category, "english"))

    return chunks

def main():
    """Main processing loop."""
    print("=== Sefaria Text Segmentation ===\n")

    sefaria_path = Path(SEFARIA_DIR)
    if not sefaria_path.exists():
        print(f"Error: {SEFARIA_DIR} not found")
        return

    json_files = list(sefaria_path.rglob("*.json"))
    print(f"Found {len(json_files)} Sefaria JSON files\n")

    all_chunks = []
    processed = 0

    for file_path in json_files:
        chunks = process_sefaria_file(file_path)
        all_chunks.extend(chunks)
        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed}/{len(json_files)} files ({len(all_chunks)} chunks)")

    print(f"\nProcessed {processed}/{len(json_files)} files")
    print(f"Total segments: {len(all_chunks)}")

    # Save to JSONL
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Statistics
    hebrew_count = sum(1 for c in all_chunks if c['language'] == 'hebrew')
    english_count = sum(1 for c in all_chunks if c['language'] == 'english')

    print(f"\n=== Statistics ===")
    print(f"Hebrew segments: {hebrew_count}")
    print(f"English segments: {english_count}")
    print(f"Total: {len(all_chunks)} segments")
    print(f"\nOutput saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()