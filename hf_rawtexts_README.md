---
license: cc-by-nc-3.0
task_categories:
- text-retrieval
- text-generation
language:
- en
- he
tags:
- judaism
- sefaria
- torah
- talmud
- hebrew
- rag
- jewish-texts
size_categories:
- 1K<n<10K
source_datasets:
- sefaria
pretty_name: Sefaria Raw Texts (judaism-llm)
---

# Sefaria Raw Texts (judaism-llm)

Raw Sefaria API responses: 6,112 JSON files, 82MB, organized by category.

**License: CC-BY-NC 3.0** — non-commercial use only. Source: [Sefaria](https://www.sefaria.org), downloaded via the official API (`download_sefaria.py` in the [pipeline repo](https://github.com/tdw419/judaism_llm)).

## Layout

Directory names encode the category path (`_` separates levels), e.g.
`Halakhah_Mishneh Torah_Commentary_Ohr Sameach_Sefer Zeraim/`. Each JSON file
is one Sefaria document with the full API schema: `text` (English),
`he` (Hebrew), `ref`, `categories`, `isComplex`, `textDepth`, etc.

### Category breakdown

| Category | Documents |
|----------|-----------|
| Halakhah | 2,088 |
| Talmud | 1,719 |
| Mishnah | 1,083 |
| Tanakh | 577 |
| Tosefta | 226 |
| Midrash | 109 |
| Responsa | 75 |
| Jewish Thought | 57 |
| Kabbalah | 48 |
| Second Temple | 46 |
| Musar | 32 |
| Chasidut | 30 |
| Liturgy | 19 |
| Reference | 3 |

## Regenerate

```bash
python3 download_sefaria.py   # from the pipeline repo
```

## Downstream

These raw files feed the RAG corpus
([tdw419/judaism-llm-rag-corpus](https://huggingface.co/datasets/tdw419/judaism-llm-rag-corpus)):
`segment_texts.py` chunks them, `embed_sefaria.py` embeds them.
