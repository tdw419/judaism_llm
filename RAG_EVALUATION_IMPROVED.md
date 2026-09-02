# RAG Evaluation Report (Improved Metrics)

## Overview

- **Total Queries:** 10
- **Grounding Score:** 30.0% (response n-grams in retrieved context)
- **Concept Coverage:** 36.8% (expected concepts in responses)
- **Average Response Length:** 1645 characters

## Test Results

| Query | Response Length | Grounding (5-gram) | Grounding (3-gram) | Is Grounded? | Concepts Covered |
|-------|----------------|-------------------|-------------------|--------------|------------------|
| What is Teshuva? | 2311 | 51.8% | 54.4% | ✓ | 3/4 |
| Explain Shabbat | 2106 | 10.9% | 18.3% | ✗ | 0/4 |
| What are the main sources of Jewish law? | 1951 | 36.9% | 40.5% | ✓ | 4/4 |
| תורה | 731 | 0.0% | 0.8% | ✗ | 0/3 |
| משנה | 1123 | 2.1% | 3.7% | ✗ | 0/3 |
| What is the significance of Rosh Hashanah? | 1953 | 34.0% | 36.5% | ✓ | 1/4 |
| Explain the concept of Kashrut | 2167 | 0.0% | 0.0% | ✗ | 4/4 |
| מה המשמעות של יום כיפור | 1238 | 0.0% | 2.8% | ✗ | 0/4 |
| פסח | 1307 | 1.8% | 3.2% | ✗ | 1/4 |
| What are the Ten Commandments? | 1560 | 0.0% | 1.3% | ✗ | 1/4 |

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

✗ RAG system poorly grounded (grounding < 40%)
✗ Concept coverage is low (coverage < 40%)

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
