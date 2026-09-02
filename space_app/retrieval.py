#!/usr/bin/env python3
"""
Hybrid retrieval for the Judaism LLM RAG.

Plain dense search with paraphrase-multilingual-MiniLM does poorly on very
short queries -- a bare word like "תורה" or "משנה" carries almost no context
for a sentence-embedding model, so its vector lands in a vague region and the
long corpus chunks it should match never surface.

retrieve() addresses that with three cheap additions on top of dense search:

  1. Query expansion -- short queries (<= 3 tokens) are also embedded wrapped
     in a few natural-language templates (Hebrew or English, picked by script),
     and the per-document best distance across all variants is pooled.
  2. Lexical recall -- for short queries every token >= 2 chars is run through
     ChromaDB's `where_document {"$contains": ...}` exact-substring filter.
     Lexical hits get a distance discount so they rank alongside dense hits.
  3. Source dedupe -- results are collapsed to one chunk per `source` so a
     single verbose commentary can't fill the whole context window.

Longer queries (> 3 tokens) fall through to ordinary dense search unchanged.
"""

import re

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_HEBREW = re.compile("[֐-׿יִ-ﭏ]")
_WORD = re.compile(r"\S+")

# Applied to short queries before embedding. "{q}" alone is always kept.
_EN_TEMPLATES = ["{q}", "What is {q}?", "Explain the concept of {q} in Judaism.",
                 "the laws of {q}"]
_HE_TEMPLATES = ["{q}", "מהי {q}?", "מה המשמעות של {q} ביהדות?", "הלכות {q}"]

SHORT_QUERY_TOKENS = 3      # <= this many tokens => expand + lexical
LEXICAL_BOOST = 0.5         # multiply distance of exact-substring hits
CANDIDATE_K = 30            # per-variant / per-token candidate pool


def _is_hebrew(text: str) -> bool:
    return bool(_HEBREW.search(text))


def _tokens(text: str):
    return _WORD.findall(text.strip())


_LETTERS = re.compile("[0-9A-Za-z֐-׿]")
_MIN_CHUNK_CHARS = 40


def _is_degenerate(doc: str) -> bool:
    """True for chunk-boundary crumbs that carry no retrievable content."""
    if doc is None:
        return True
    stripped = re.sub(r"&[a-z]+;|\s+", "", doc)
    if len(stripped) < _MIN_CHUNK_CHARS:
        return True
    return len(_LETTERS.findall(doc)) < 15


def expand_query(query: str):
    """Return the list of query strings to embed (>=1, first is the original)."""
    q = query.strip()
    if len(_tokens(q)) > SHORT_QUERY_TOKENS:
        return [q]
    templates = _HE_TEMPLATES if _is_hebrew(q) else _EN_TEMPLATES
    variants, seen = [], set()
    for t in templates:
        v = t.format(q=q)
        if v not in seen:
            seen.add(v)
            variants.append(v)
    return variants


def retrieve(query, collection, embedding_model, top_k=5,
             candidate_k=CANDIDATE_K):
    """Hybrid dense + lexical retrieval.

    Returns the same shape ChromaDB's .query() does (lists nested one level):
    {"ids": [[...]], "documents": [[...]], "metadatas": [[...]],
     "distances": [[...]]}
    """
    variants = expand_query(query)
    variant_embs = embedding_model.encode(
        variants, convert_to_numpy=True, normalize_embeddings=True
    )
    if variant_embs.ndim == 1:
        variant_embs = variant_embs.reshape(1, -1)

    short_query = len(_tokens(query)) <= SHORT_QUERY_TOKENS

    # id -> [best_distance, document, metadata, lexical_hit]
    pooled = {}

    def _absorb(res, scale=1.0, lexical=False):
        if not res["ids"] or not res["ids"][0]:
            return
        for i, _id in enumerate(res["ids"][0]):
            doc = res["documents"][0][i]
            # Drop degenerate chunks (chunk-boundary crumbs like ':' or
            # '&nbsp;'). They embed near the origin and spuriously match
            # low-context short queries.
            if _is_degenerate(doc):
                continue
            dist = res["distances"][0][i] * scale
            if _id not in pooled:
                pooled[_id] = [dist, doc, res["metadatas"][0][i], lexical]
            else:
                if dist < pooled[_id][0]:
                    pooled[_id][0] = dist
                pooled[_id][3] = pooled[_id][3] or lexical

    # 1 + 2a: dense search for every (expanded) query variant
    for emb in variant_embs:
        _absorb(collection.query(query_embeddings=[emb.tolist()],
                                 n_results=candidate_k))

    # 2b: lexical exact-substring recall for short queries
    query_tokens = [t for t in _tokens(query) if len(t) >= 2]
    if short_query:
        for tok in query_tokens:
            try:
                res = collection.query(
                    query_embeddings=[variant_embs[0].tolist()],
                    n_results=candidate_k,
                    where_document={"$contains": tok},
                )
            except Exception:
                continue
            _absorb(res, scale=LEXICAL_BOOST, lexical=True)

    # 3: rank, then dedupe by source. For short queries an exact-substring
    # (lexical) hit always outranks a pure-dense hit -- a bare word like
    # "משנה" is a keyword lookup more than a semantic one.
    def _key(kv):
        dist, _doc, _meta, lexical = kv[1]
        if short_query:
            return (0 if lexical else 1, dist)
        return (0, dist)

    ranked = sorted(pooled.items(), key=_key)
    ids, docs, metas, dists, seen_src = [], [], [], [], set()
    for _id, (dist, doc, meta, _lexical) in ranked:
        src = meta.get("source")
        if src in seen_src:
            continue
        seen_src.add(src)
        ids.append(_id)
        docs.append(doc)
        metas.append(meta)
        dists.append(dist)
        if len(ids) >= top_k:
            break

    return {"ids": [ids], "documents": [docs],
            "metadatas": [metas], "distances": [dists]}
