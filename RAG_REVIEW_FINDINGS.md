# RAG System Review: Findings and Fixes

## Review Summary

A comprehensive review identified critical bugs in the RAG system that invalidated the "production-ready" claim. This document tracks findings, fixes applied, and remaining issues.

## Critical Issues (Fixed)

### 1. Vector DB stores chunk_id, not text ✓ FIXED
- **Issue:** run_phase2.py stored chunk_id strings (e.g., "Orot HaTeshuvah_1_1") instead of actual segment text
- **Impact:** Retrieval returned titles only; RAG responses were not grounded in context
- **Fix:** Modified line 83 to index `segments[j]['text']` instead of chunk_id
- **Verification:** 3/3 retrieved documents contain actual text (2048 chars each)
- **Commit:** c60da22

### 2. Citations were hallucinated ✓ ADDRESSED
- **Issue:** Model invented quotes and citations it never saw in context
- **Root Cause:** Related to #1 - no text in context = no grounding
- **Mitigation:** Fixed #1 reduces hallucinations; added grounding metrics
- **Remaining:** Need improved evaluation to measure this

### 3. Evaluation metrics don't measure retrieval ✓ IMPROVED
- **Issue:** source_accuracy only matched book names, not text usage
- **Fix:** Created run_phase6_improved.py with:
  - Grounding Score (n-gram overlap)
  - Concept Coverage (expected concepts in responses)
- **Commit:** f0d84eb

## High-Priority Issues (Noted, Not Fixed)

### 4. .gitignore excludes critical files
- **Issue:** segments.jsonl, *.safetensors excluded
- **Impact:** Fresh clone cannot reproduce or run
- **Status:** Noted, requires .gitignore changes and large file storage strategy

### 5. Docker deployment broken
- **Issues:**
  - MODEL_PATH mismatch (hardcoded vs ENV)
  - Missing nginx.conf
  - No .dockerignore (huge image)
  - Obsolete docker-compose version
- **Status:** Noted, requires Dockerfile overhaul

### 6. CORSMiddleware configuration invalid
- **Issue:** allow_origins=["*"] + allow_credentials=True rejected by spec
- **Status:** Noted, requires CORS config fix

### 7. HTML not stripped from segments
- **Issue:** HTML tags stored in embeddings and context
- **Status:** Noted, requires stripping in run_phase1.py

### 8. Sefaria schema loss
- **Issue:** Complex list-of-lists dropped, losing unknown fraction of content
- **Status:** Noted, requires improved parsing in run_phase1.py

### 9. Chunking mid-word/mid-verse
- **Issue:** Fixed 2048-byte cuts split words/verses
- **Status:** Noted, requires boundary-aware chunking

### 10. Non-reproducible eval
- **Issue:** No seed set; results vary between runs
- **Status:** Noted, requires seed in model.generate()

### 11. Phase 4 unverified
- **Issue:** Interactive CLI never tested
- **Status:** Noted, requires manual testing

### 12. Deprecated API usage
- **Issue:** @app.on_event("startup") deprecated in FastAPI
- **Status:** Noted, requires lifespan event

### 13. No query length bound
- **Issue:** /query has no input length limit
- **Status:** Noted, requires input validation

## What Actually Works

- Phase 1/2 mechanics: 18,453 segments → ChromaDB with 18,453 rows
- Multilingual embedding choice (paraphrase-multilingual-MiniLM-L12-v2)
- FastAPI + CLI + eval scaffolding
- Merged Qwen2.5-7B model is present and complete

## RAG System Status After Fixes

**Fixed Issues (Critical):**
1. ✓ Vector DB now stores actual text
2. ✓ Retrieval returns passages, not titles
3. ✓ Improved evaluation metrics (grounding, concept coverage)

**System Quality:**
- Retrieval: Now functional (text-based)
- Grounding: Can be measured (n-gram overlap)
- Citations: Less likely to hallucinate (but not fully verified)

**Remaining Work:**
- ✓ Improved evaluation complete (10 queries, commit 9a7db66)
- Address high-priority issues (#4-13)
- Update .gitignore and deployment configs

## Metrics (Final Results)

After improved evaluation with 10 queries:

- **Grounding Score:** 30.0% (response n-grams in retrieved context)
- **Concept Coverage:** 36.8% (expected concepts in responses)
- **Average Response Length:** 1,645 characters
- **Grounded Queries:** 3/10

**Grounded Queries:**
- What is Teshuva? ✓ (51.8% 5-gram, 3/4 concepts)
- Main sources of Jewish law? ✓ (36.9% 5-gram, 4/4 concepts)
- Rosh Hashanah? ✓ (34.0% 5-gram, 1/4 concepts)

**Not Grounded (7/10):**
- Explain Shabbat (10.9% 5-gram) - Beitzah commentary irrelevant
- Torah (0.0% 5-gram) - single-word query fails
- Mishnah (2.1% 5-gram) - single-word query fails
- Kashrut (0.0% 5-gram, 4/4 concepts) - model knowledge, no context use
- Yom Kippur (0.0% 5-gram) - single-word query fails
- Pesach (1.8% 5-gram) - single-word query fails
- Ten Commandments (0.0% 5-gram) - retrieval fails

**Additional Fixes Applied:**
- HTML stripping (commit 94dd471) - Improved English segments (450→5,176)
- Expanded eval set (5→10 queries)
- Removed invalid seed kwarg

**Corpus Impact (HTML stripped):**
- Hebrew segments: 11,614 (↓ from 16,903)
- English segments: 5,176 (↑ from 450)
- Total: 16,790 (↓ from 18,453)

## System Quality (Final)

**Retrieval:** Functional but poor quality (30% grounding)
**Grounding:** Measurable but low (addressable with retrieval improvements)
**Citations:** Less likely to hallucinate (but not fully verified)

**Status:** RAG system works but retrieval quality needs improvement

## Next Steps

1. Increase top_k (5→10) to improve retrieval
2. Improve embedding model quality
3. Query expansion for single-word Hebrew queries
4. Boundary-aware chunking (mid-word cuts destroy coherence)
5. Fix Docker deployment (#5)
6. Update .gitignore (#4)

## Conclusions

The RAG system is functionally working but not "production-ready." Critical retrieval bug (#1) has been fixed, and evaluation shows 30% grounding with 3/10 queries grounded. HTML stripping improved the corpus but didn't fix core retrieval quality.

**Status: Working prototype, needs retrieval quality improvements**

**Next Steps:**
1. Improve retrieval (increase top_k, better embeddings, query expansion)
2. Address .gitignore and reproduction issues
3. Fix Docker deployment
4. Validate citation accuracy with real examples

---

**Review Date:** September 1, 2026
**Fixes Applied:** 2 commits (c60da22, f0d84eb)
**Open Issues:** 10 high-priority items noted