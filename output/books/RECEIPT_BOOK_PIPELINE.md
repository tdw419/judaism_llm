# Receipt: RAG-grounded Hebrew book generation (עשיית צדק)

Date: 2026-09-11 · Branch: roadmap-execution · GPU: RTX 5090 Laptop 24GB

## What was built

`write_book.py` — outline-driven book generator on top of the existing RAG
stack (retrieval.py hybrid search + ChromaDB `sefaria_texts`, 16,790 chunks,
Sefaria corpus). Modes:

- Default: retrieval → LLM editor's note → citation gate → verbatim passages.
- `--no-notes`: deterministic sourcebook, no model load, passages only.

Citation gate (`verify_note`): every "..." quote in an editor's note must
appear verbatim (quote/case/space-normalized) in the passage cited as [N].
Language gate: ≥85% Hebrew letters, zero CJK; junk notes are dropped to
passage-only. One retry per note.

## Iteration history (all runs verified, not assumed)

1. v1 essay mode (output/books/practicing_righteousness.md): model wove its
   own quotes into essays. Gate: only 4/~80 quotes verified — invented
   citations, quote-loops, mixed languages. REJECTED by design of gate.
2. v2 sourcebook + notes (practicing_righteousness_v2.md): body now verbatim
   attributed sources (correct), but gate passed 2 notes with Chinese/English
   drift — language check counted Hebrew chars, not Hebrew ratio. Tightened.
3. v3 (practicing_righteousness_v3.md): strict gate dropped 6/9 notes;
   survivors still carried artifacts (×</p>, latin transliterations).
   Conclusion: this 7B fine-tune (trained on raw source text, not editorial
   prose) cannot write clean Hebrew commentary. Kept gate, exposed
   `--no-notes`.
4. FINAL (practicing_righteousness_final.md): `--no-notes` sourcebook.
   9 sections × 5 passages = 45 verbatim attributed Sefaria sources,
   241 lines / 138KB, no duplication.

## Bug found and fixed along the way

- Incremental save wrote the full book each section; the footer then
  appended the book list a second time → every output file was duplicated
  (grep '^###' showed 18 sections in a 9-section book). Fixed: footer only
  appended. Verified counts after fix: 9 '###', 45 passage headers.
- OOM on first run: idle Ollama qwen2.5-coder:14b runner held 14.6GB VRAM.
  Unloaded via `/api/generate` with `keep_alive:0`. Ollama reloads on demand.

## Honest capability statement

- GOOD: sourcebook assembly — outline + hybrid retrieval + verbatim,
  attributed passages. Fully deterministic, zero hallucination surface.
- LIMITED: LLM-written Hebrew editor's notes — gate keeps only the honest
  ones and the model fails it most of the time. Do not trust unsupervised
  notes from this checkpoint.
- NOT for: novel halakha / psak. Disclaimer is baked into the front matter.

## Rerun

```bash
python3 write_book.py --outline outline_practicing_righteousness.json \
    --out output/books/<name>.md --no-notes        # deterministic
# drop --no-notes to attempt editor's notes (gate applies)
```
