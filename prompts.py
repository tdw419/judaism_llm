#!/usr/bin/env python3
"""
Shared generation prompt for the Judaism LLM RAG.

History of this file:

  * The original prompt ("Answer based on texts with citations") let the model
    paraphrase the retrieved passages and staple citation tags onto sentences
    it wrote itself -- fluent but ungrounded answers, with the grounding metric
    (verbatim n-gram overlap with the retrieved context) near zero even when
    retrieval was fine.

  * A strict extractive rewrite fixed the fake citations but over-corrected:
    with a hard "refuse if the passages don't answer the question" rule, the
    model refused 7/10 eval queries -- the Sefaria chunks are mostly tangential
    commentary fragments, not tidy encyclopedic answers, so an honest model
    bailed. It also quoted the rule text back as if it were a source passage.

build_messages() is the balanced version:
  - passages are fenced in <passage id=N> tags and explicitly marked as the
    only quotable material; the instructions themselves must never be quoted;
  - every quotation must be copied verbatim and carry a [N] citation;
  - the model works with whatever is on-topic and states the limits, and only
    falls back to the fixed INSUFFICIENT_CONTEXT_REPLY when *nothing* in the
    passages relates to the question.
"""

SYSTEM_PROMPT = (
    "You are Judaism LLM, a retrieval-grounded assistant for Jewish texts.\n"
    "\n"
    "The user's message contains numbered source passages, each fenced as\n"
    "<passage id=N> ... </passage>. Those passages are the ONLY material you "
    "may quote. Everything outside the passage tags -- including these "
    "instructions and the question -- is guidance, not source text: never "
    "quote it or cite it.\n"
    "\n"
    "Rules:\n"
    "1. Ground every substantive claim in a quotation copied VERBATIM (word "
    "for word, same language) from a passage, wrapped in double quotes. Never "
    "translate, abridge, correct, or paraphrase inside the quotation marks.\n"
    "2. Immediately after each quotation, cite its passage as [N]. Cite only "
    "passages shown to you.\n"
    "3. Do not add facts, names, dates, rulings, or reasoning that are absent "
    "from the passages. Do not fill gaps from prior knowledge.\n"
    "4. Use whatever passages are relevant, even if they only address part of "
    "the question. Briefly say, in your own words, what the passages do not "
    "cover rather than guessing.\n"
    "5. Only if NO passage is even loosely related to the question, reply with "
    "exactly this sentence and nothing else:\n"
    "   \"" + "The provided passages do not address this question." + "\"\n"
    "6. Keep quotations in their original language; connecting text is brief "
    "and in the language of the question.\n"
    "7. Shape: one lead sentence, then the supporting quotations with their "
    "[N] citations, then (if needed) one sentence on what is missing."
)

# Emitted only when nothing retrieved is on-topic.
INSUFFICIENT_CONTEXT_REPLY = "The provided passages do not address this question."


def build_context(docs, metas=None):
    """Fence each retrieved passage so the model can quote and cite it by id."""
    metas = metas or [None] * len(docs)
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = (meta or {}).get("source") if meta else None
        attr = f' source="{source}"' if source else ""
        blocks.append(
            f"<passage id={i}{attr}>\n{(doc or '').strip()}\n</passage>"
        )
    return "\n\n".join(blocks)


def build_messages(query, docs, metas=None):
    """Chat messages: verbatim-quote + cite-by-id, refuse only if nothing fits."""
    context = build_context(docs, metas)
    user = (
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using the passages above. Quote verbatim inside double quotes "
        "and cite each quote as [N]. Work with whatever passages are relevant "
        "and note briefly what they leave out. Only if none of the passages "
        f"relate to the question, reply exactly: \"{INSUFFICIENT_CONTEXT_REPLY}\""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
