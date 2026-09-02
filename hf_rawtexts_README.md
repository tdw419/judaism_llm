# Sefaria Raw Texts (judaism-llm)

Raw Sefaria API responses: 6,112 JSON files, 82MB, organized by category.

**License: CC-BY-NC 3.0** — non-commercial use only. Source: [Sefaria](https://www.sefaria.org), downloaded via the official API (`download_sefaria.py` in the [pipeline repo](https://github.com/tdw419/judaism_llm)).

## Layout

Directory names encode the category path (`_` separates levels), e.g.
`Halakhah_Mishneh Torah_Commentary_Ohr Sameach_Sefer Zeraim/`. Each JSON file
is one Sefaria document with the full API schema: `text` (English),
`he` (Hebrew), `ref`, `categories`, `isComplex`, `textDepth`, etc.

## Regenerate

```bash
python3 download_sefaria.py   # from the pipeline repo
```

## Downstream

These raw files feed the RAG corpus
([tdw419/judaism-llm-rag-corpus](https://huggingface.co/datasets/tdw419/judaism-llm-rag-corpus)):
`segment_texts.py` chunks them, `embed_sefaria.py` embeds them.
