# RAG Evaluation Report (Improved Metrics)

## Overview

- **Total Queries:** 10
- **Grounding Score:** 30.0% (response n-grams in retrieved context)
- **Concept Coverage:** 31.6% (expected concepts in responses)
- **Average Response Length:** 1790 characters

## Test Results

| Query | Response Length | Grounding (5-gram) | Grounding (3-gram) | Is Grounded? | Concepts Covered |
|-------|----------------|-------------------|-------------------|--------------|------------------|
| What is Teshuva? | 2010 | 49.8% | 54.5% | ✓ | 3/4 |
| Explain Shabbat | 2378 | 2.5% | 10.3% | ✗ | 0/4 |
| What are the main sources of Jewish law? | 1750 | 25.0% | 30.6% | ✓ | 3/4 |
| תורה | 1762 | 0.0% | 0.0% | ✗ | 0/3 |
| משנה | 2121 | 0.0% | 0.0% | ✗ | 0/3 |
| What is the significance of Rosh Hashanah? | 2401 | 19.0% | 23.2% | ✗ | 2/4 |
| Explain the concept of Kashrut | 1927 | 0.0% | 0.0% | ✗ | 3/4 |
| מה המשמעות של יום כיפור | 1113 | 31.0% | 35.8% | ✓ | 0/4 |
| פסח | 1084 | 18.7% | 24.3% | ✗ | 0/4 |
| What are the Ten Commandments? | 1359 | 0.0% | 3.9% | ✗ | 1/4 |

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
