# RAG System Roadmap: Sefaria + Judaism LLM

This roadmap creates a RAG (Retrieval-Augmented Generation) system combining Sefaria texts with the fine-tuned Judaism LLM model.

## Architecture Overview

```
User Query → Embed → Vector Search (ChromaDB) → Top K Passages → Judaism LLM → Response + Sources
```

## Phase 1: Data Preparation (Day 1)

### Tasks

- [ ] **Create embedding pipeline**
  - Install `sentence-transformers` (multilingual m3e-base-x)
  - Test Hebrew/English embedding quality
  - Script: `embed_sefaria.py`
  - Target: Embed all 6,112 Sefaria texts

- [ ] **Segment texts for retrieval**
  - Split long texts into 512-token chunks
  - Preserve structure (chapter/verse references)
  - Overlap chunks (50 tokens) for context continuity
  - Script: `segment_texts.py`
  - Target: ~50,000 segments

**Acceptance Criteria:**
- All Sefaria texts embedded and stored
- Segments maintain Hebrew/English pairing
- Total segments: >40,000

---

## Phase 2: Vector Database (Day 1-2)

### Tasks

- [ ] **Set up ChromaDB**
  - Install `chromadb`
  - Create collection: `sefaria_texts`
  - Configure: persist directory, embedding function
  - Script: `setup_chroma.py`

- [ ] **Index Sefaria embeddings**
  - Load segments from Phase 1
  - Batch insert into ChromaDB (1000 at a time)
  - Store metadata: source, category, language
  - Target: All 50,000+ segments indexed

- [ ] **Test retrieval**
  - Query: "What is Teshuva?"
  - Verify: Returns relevant Hebrew/English passages
  - Check: Metadata accuracy (source, verse)
  - Script: `test_retrieval.py`

**Acceptance Criteria:**
- ChromaDB collection: `sefaria_texts`
- Indexed: >40,000 segments
- Retrieval accuracy: Top 5 relevant for test queries

---

## Phase 3: RAG Query Engine (Day 2-3)

### Tasks

- [ ] **Create RAG pipeline**
  - Script: `rag_query.py`
  - Components:
    1. Query embedding
    2. Vector search (top_k=5)
    3. Context assembly
    4. Generation with Judaism LLM
    5. Source citation formatting

- [ ] **Implement context assembly**
  - Format retrieved passages with metadata
  - Hebrew + English passages in same context
  - Character limit: 2048 tokens (model limit)

- [ ] **Integrate Judaism LLM**
  - Load: `judaism-llm-qwen2.5-7b-merged`
  - Prompt template:
    ```
    Based on these Sefaria texts:
    [PASSAGES WITH SOURCES]

    Answer the question: [USER QUERY]
    Include citations from the passages.
    ```

- [ ] **Add source citations**
  - Extract metadata from retrieved segments
  - Format: `[Source: Tractate, Chapter:Verse]`
  - Language detection (Hebrew/English)

**Acceptance Criteria:**
- RAG query returns response + sources
- Response uses retrieved context
- Citations accurate to source texts

---

## Phase 4: Interactive CLI (Day 3-4)

### Tasks

- [ ] **Create interactive CLI**
  - Script: `rag_chat.py`
  - Features:
    1. Hebrew/English input
    2. Real-time retrieval + generation
    3. Source display
    4. Conversation history (optional)

- [ ] **Add command-line arguments**
  ```bash
  python3 rag_chat.py --model judaism-llm-qwen2.5-7b-merged --collection sefaria_texts --top_k 5
  ```

- [ ] **Improve UX**
  - Color output (Hebrew: cyan, sources: yellow)
  - Latency display (retrieval + generation time)
  - Error handling (empty retrieval)

**Acceptance Criteria:**
- Interactive chat with Hebrew/English support
- Each response includes 5 sources
- Total latency < 10 seconds

---

## Phase 5: Web Interface (Optional, Day 4-5)

### Tasks

- [ ] **Create FastAPI backend**
  - Endpoint: `POST /query`
  - Input: `{ "query": "What is Shabbat?", "top_k": 5 }`
  - Output: `{ "response": "...", "sources": [...] }`

- [ ] **Create frontend**
  - Simple HTML/JS interface
  - Query input box
  - Response + sources display
  - Language toggle (Hebrew/English)

**Acceptance Criteria:**
- FastAPI server running on port 8000
- Frontend accessible at `http://localhost:8000`
- API returns response + sources in < 5 seconds

---

## Phase 6: Evaluation (Day 5)

### Tasks

- [ ] **Create evaluation suite**
  - Test queries: 50 (25 Hebrew, 25 English)
  - Metrics:
    1. Retrieval accuracy (MRR, NDCG@5)
    2. Response quality (human evaluation)
    3. Source citation accuracy
  - Script: `evaluate_rag.py`

- [ ] **Compare baselines**
  - Baseline 1: Judaism LLM without RAG
  - Baseline 2: Judaism LLM with random passages
  - Target: RAG outperforms baselines

- [ ] **Document results**
  - Write: `RAG_EVALUATION.md`
  - Include: Metrics, sample outputs, limitations

**Acceptance Criteria:**
- Retrieval accuracy: MRR > 0.7
- 80% of responses have accurate sources
- Evaluation report complete

---

## Phase 7: Deployment (Day 6)

### Tasks

- [ ] **Package for deployment**
  - Docker container: `Dockerfile`
  - Requirements: `requirements.txt`
  - Environment variables: `MODEL_PATH`, `CHROMA_PATH`

- [ ] **Deploy to production**
  - Deploy ChromaDB (persistent storage)
  - Deploy FastAPI (auto-scaling)
  - Monitor: retrieval latency, error rate

**Acceptance Criteria:**
- Docker image built
- Service running in production
- Monitoring dashboard active

---

## Technical Decisions

### Embedding Model
**Choice:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reason: Strong multilingual (Hebrew + English), 384-dim (fast)
- Alternative: `intfloat/multilingual-e5-large` (higher quality, slower)

### Vector Database
**Choice:** ChromaDB
- Reason: Local, no external dependencies, good multilingual support
- Alternative: Qdrant (cloud, better scaling)

### Retrieval Strategy
**Choice:** Semantic search with top_k=5
- Reason: Simple, effective for Sefaria texts
- Future: Hybrid (semantic + keyword search)

### Generation Model
**Choice:** `judaism-llm-qwen2.5-7b-merged` (already trained)
- Reason: Already fine-tuned on Sefaria, bilingual support
- Context window: 2048 tokens (sufficient for 5 passages)

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Retrieval latency | < 2 seconds |
| Generation latency | < 5 seconds |
| Total latency | < 10 seconds |
| Retrieval accuracy (MRR) | > 0.7 |
| Source citation accuracy | > 80% |
| Memory usage (ChromaDB) | < 10GB |
| Memory usage (Model) | < 15GB |

---

## File Structure

```
judaism_llm/
├── rag/
│   ├── embed_sefaria.py       # Phase 1
│   ├── segment_texts.py       # Phase 1
│   ├── setup_chroma.py        # Phase 2
│   ├── test_retrieval.py      # Phase 2
│   ├── rag_query.py           # Phase 3
│   ├── rag_chat.py            # Phase 4
│   ├── evaluate_rag.py        # Phase 6
│   └── chroma_db/             # ChromaDB data directory
├── judaism-llm-qwen2.5-7b-merged/  # Trained model
└── RAG_ROADMAP.md             # This file
```

---

## Next Steps

1. Start Phase 1: `python3 embed_sefaria.py`
2. Verify embeddings: Test Hebrew/English similarity
3. Continue Phase 2-7 sequentially

---

**Created:** September 1, 2026
**Branch:** roadmap-execution
**Status:** Not started