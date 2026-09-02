---
title: Judaism LLM — Sefaria RAG
emoji: 🕍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# Judaism LLM — Sefaria RAG

Retrieval-grounded chat over 16,790 Sefaria segments using
[tdw419/judaism-llm-qwen2.5-7b](https://huggingface.co/tdw419/judaism-llm-qwen2.5-7b).

Answers are **verbatim quotes** from retrieved passages with [N] citations,
or an honest refusal when nothing retrieved is relevant. Pipeline code:
https://github.com/tdw419/judaism_llm

Corpus: https://huggingface.co/datasets/tdw419/judaism-llm-rag-corpus
(CCs BY-NC 3.0 via Sefaria — non-commercial use only)
