# Sefaria RAG Corpus (judaism-llm)

Retrieval corpus for the [judaism-llm RAG pipeline](https://github.com/tdw419/judaism_llm):
16,790 text segments from Sefaria with pre-computed multilingual embeddings.

## Files

| File | Size | Description |
|------|------|-------------|
| `segments.jsonl` | 45 MB | 16,790 segments: `{text, source, category, language, chunk_id}` |
| `sefaria_embeddings.npy` | 25 MB | float32 array, shape (16790, 384), row `i` = embedding of line `i` of segments.jsonl |
| `sefaria_metadata.jsonl` | 3 MB | Per-segment metadata: `{index, chunk_id, source, category, language}` |

- **Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, normalized)
- **Languages:** 11,614 Hebrew, 5,176 English segments
- **Chunking:** 2048-char windows (note: hard cuts, mid-word splits possible — sentence-aware re-chunking is planned)

## Usage

```python
import json
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

repo = "tdw419/judaism-llm-rag-corpus"
segments = [json.loads(l) for l in open(hf_hub_download(repo, "segments.jsonl", repo_type="dataset"), encoding="utf-8")]
embeddings = np.load(hf_hub_download(repo, "sefaria_embeddings.npy", repo_type="dataset"))

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
query_emb = model.encode("What is Teshuva?", normalize_embeddings=True)
scores = embeddings @ query_emb
top = scores.argsort()[::-1][:5]
for i in top:
    print(segments[i]["source"], ":", segments[i]["text"][:100])
```

## Provenance & License

- Texts from [Sefaria](https://www.sefaria.org) (official API download)
- **License: CC-BY-NC 3.0** (Sefaria's terms) — non-commercial use only

## Companion

- Model: [tdw419/judaism-llm-qwen2.5-7b](https://huggingface.co/tdw419/judaism-llm-qwen2.5-7b)
- Pipeline code (retrieval, extractive prompt, eval): https://github.com/tdw419/judaism_llm
