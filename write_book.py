#!/usr/bin/env python3
"""
write_book.py -- RAG-grounded Hebrew sourcebook generator for Judaism LLM.

Format: sourcebook (קובץ מקורות). Pilot v1 taught us the 7B model cannot be
trusted to weave its own verbatim quotes into a Hebrew essay -- the citation
gate failed ~90% of its quotes (invented citations, garbled quoting, loops).
So v2 flips the genre to what Jewish learning actually uses: the retrieved
Sefaria passages ARE the body, printed verbatim with attribution, and the
model writes only a short "editor's note" (הערת העורך) introducing them.
The gate still checks every quote inside the editor's note; failing notes
are regenerated once, then stripped to passage-only if it fails again.

Usage:
  python3 write_book.py --outline outline_practicing_righteousness.json \
      --out output/books/practicing_righteousness.md [--max-sections N]
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from retrieval import retrieve  # noqa: E402

MODEL_PATH = os.path.join(HERE, "judaism-llm-qwen2.5-7b-merged")
MAX_NEW_TOKENS = 220          # editor's note only -- short
TEMPERATURE = 0.3             # low: notes are connective tissue, not poetry
REPEAT_PENALTY = 1.1
NO_REPEAT_NGRAM = 6           # hard stop for the quote-loop degeneracy
TOP_K = 5                     # passages printed per section

DISCLAIMER_HE = (
    "> **הערה חשובה:** ספר זה נערך באופן אוטומטי בידי Judaism LLM "
    "(Qwen2.5-7B) מתוך מאגר ספריא. גוף הספר הוא ציטוטי מקור מילה במילה "
    "כפי שהוחזרו מהמאגר; הערות העורך נכתבו בידי המודל ונבדקו בשער אימות "
    "ציטוטים (כל ציטוט בהערה חייב להופיע מילה במילה במקור שאוחזר). "
    "אין בספר זה כדי פסיקה הלכתית, ואין להסתמך עליו להכרעות מעשיות. "
    "האחזור עשוי להחסיר מקורות מרכזיים; ללימוד רציני יש לחזור אל המקורות "
    "בעיון ישיר."
)

EDITOR_SYSTEM = (
    "אתה עורך ספרותי של קובץ מקורות בעברית. קיבלת קטעי מקור מסופריא "
    "בתגיות <passage id=N>. כתוב הערת עורך קצרה (2-4 משפטים, בעברית בלבד) "
    "המציגה את הנושא ומסבירה מה הקטעים למדים ממנו. אם אתה מצטט -- צטט מילה "
    "במילה מתוך הקטעים בלבד וסמן [N]. אל תמציא מקורות, שמות או עובדות "
    "שאינם בקטעים. אל תכתוב באנגלית."
)

_EDITOR_USER_TMPL = (
    "{context}\n\n"
    "נושא הסעיף: {title} (שאילתת האחזור: {query})\n\n"
    "כתוב הערת עורך קצרה בעברית לסעיף זה, המבוססת על הקטעים למעלה."
)

_QUOTE_RE = re.compile(r"\u201c([^\u201d]{10,})\u201d|\"([^\"]{10,})\"")
_CITE_RE = re.compile(r"\[(\d+)\]")


def load_collection():
    import chromadb

    client = chromadb.PersistentClient(path=os.path.join(HERE, "chroma_db"))
    return client.get_collection("sefaria_texts")


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)
    return model, tok


def generate(model, tok, system, user):
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            repetition_penalty=REPEAT_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def _norm(s):
    """Quote-agnostic normalization for verbatim matching."""
    s = re.sub(r"[\u201c\u201d\"'\u05f4\u05f3]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def verify_note(note: str, docs):
    """Check every "..." quote in the note appears verbatim in a cited doc.

    Returns (ok, clean_note). Uncited or unverified quotes are removed
    (leaving the surrounding prose). A note whose prose is then mostly holes
    should be rejected by the caller (ok=False) and retried.
    """
    n_docs = len(docs)
    bad_spans = []
    for m in _QUOTE_RE.finditer(note):
        quote = m.group(1) if m.group(1) is not None else m.group(2)
        tail = note[m.end():m.end() + 120]
        cm = _CITE_RE.search(tail)
        ok = False
        if cm and 1 <= int(cm.group(1)) <= n_docs:
            doc = docs[int(cm.group(1)) - 1]
            ok = _norm(quote) in _norm(doc)
        if not ok:
            bad_spans.append(m.span())
    if not bad_spans:
        return True, note

    # strip bad quotes (keep everything else)
    out, last = [], 0
    for a, b in bad_spans:
        out.append(note[last:a])
        last = b
    out.append(note[last:])
    clean = re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
    return False, clean


def editor_note(model, tok, title, query, docs, metas):
    """Generate + gate the editor's note. One retry; else passage-only."""
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = (meta or {}).get("source", "unknown")
        blocks.append(f'<passage id={i} source="{src}">\n{doc.strip()}\n'
                      f"</passage>")
    user = _EDITOR_USER_TMPL.format(context="\n\n".join(blocks),
                                    title=title, query=query)

    note = generate(model, tok, EDITOR_SYSTEM, user)
    ok, clean = verify_note(note, docs)
    if not ok:
        # one retry, then strip any remaining quotes and keep the prose
        note2 = generate(model, tok, EDITOR_SYSTEM, user)
        ok, clean = verify_note(note2, docs)
        if not ok:
            clean = re.sub(r"\u201c[^\u201d]{10,}\u201d|\"[^\"]{10,}\"",
                           "", clean)
    # hard language sanity: the note must be almost entirely Hebrew.
    # Count letters only (CJK/latin letters vs Hebrew letters); notes that
    # drift into Chinese/English mid-stream get dropped, passage-only wins.
    heb = len(re.findall(r"[\u0590-\u05FF]", clean or ""))
    latin = len(re.findall(r"[A-Za-z]", clean or ""))
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", clean or ""))
    total_letters = heb + latin + cjk
    if (clean and (cjk > 0 or total_letters == 0
                   or heb / total_letters < 0.85)) or len(clean) < 30:
        clean = ""
    # strip redundant self-labels the model likes to emit
    clean = re.sub(r"^הערת\s*(ה\s*)?עורך\s*[:：]?\s*", "", clean or "").strip()
    clean = re.sub(r"^הערה\s*[:：]\s*", "", clean).strip()
    return clean


def section_body(docs, metas):
    lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = (meta or {}).get("source", "unknown")
        lines.append(f"**{i}. {src}**")
        lines.append("")
        lines.append(doc.strip())
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sections", type=int, default=0)
    ap.add_argument("--no-notes", action="store_true",
                    help="deterministic sourcebook: skip the LLM entirely "
                         "(no model load) -- verbatim passages only")
    args = ap.parse_args()

    with open(args.outline, encoding="utf-8") as f:
        outline = json.load(f)

    col = load_collection()
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = tok = None
    if not args.no_notes:
        model, tok = load_model()

    book = [f"# {outline['title']}", "", DISCLAIMER_HE, ""]
    if outline.get("intro"):
        book += [outline["intro"], ""]

    total = sum(len(ch["sections"]) for ch in outline["chapters"])
    done, t0 = 0, time.time()

    for chapter in outline["chapters"]:
        book += [f"## {chapter['title']}", ""]
        for sec in chapter["sections"]:
            if args.max_sections and done >= args.max_sections:
                break
            title, query = sec["title"], sec["query"]
            print(f"\n=== [{done+1}/{total}] {title}", flush=True)

            res = retrieve(query, col, emb, top_k=TOP_K)
            docs, metas = res["documents"][0], res["metadatas"][0]
            srcs = [(m or {}).get("source", "?") for m in metas]
            print(f"    retrieved: {', '.join(s[:40] for s in srcs[:3])}…",
                  flush=True)

            t_gen = time.time()
            note = ""
            if model is not None:
                note = editor_note(model, tok, title, query, docs, metas)
            body = section_body(docs, metas)
            print(f"    note: {len(note)} chars, gated "
                  f"({time.time()-t_gen:.0f}s) | passages: {len(docs)}",
                  flush=True)

            book += [f"### {title}", ""]
            if note:
                book += [f"**הערת העורך:** {note}", ""]
            book += [body, "---", ""]
            done += 1

            out_path = os.path.join(HERE, args.out)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(book))
        if args.max_sections and done >= args.max_sections:
            break

    elapsed = time.time() - t0
    footer = ["---",
              f"_נערך אוטומטית: {done} סעיפים · {elapsed/60:.1f} דקות · "
              "מודל: Judaism-LLM (Qwen2.5-7B) · אחזור: Sefaria/ChromaDB · "
              "כל ציטוט בהערות העורך אומת מול המקור האחזור_"]
    book += footer
    with open(os.path.join(HERE, args.out), "a", encoding="utf-8") as f:
        f.write("\n".join(footer) + "\n")
    print(f"\nBook written: {args.out} ({done} sections, "
          f"{elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
