#!/usr/bin/env python3
"""Equidistant Letter Sequence (ELS) search over the Torah — "Bible codes".

DISCLAIMER: This is a recreational text-analysis toy, not a research tool.
The 1994 Witztum-Rips "Great Rabbis" ELS experiment (Statistical Science)
claimed statistically significant hidden encodings in the Torah. Brendan
McKay, Dror Bar-Natan, Maya Bar-Hillel, and Gil Kalai (Statistical Science,
1999) showed the same methodology finds equally "significant" matches in
Hebrew translations of Moby Dick and War and Peace, and that undisclosed
researcher discretion in choosing word spellings/forms drove the original
result. There is no credible evidence ELS matches encode real information
beyond what chance predicts in any long text. Treat all output here as a
puzzle-generator, not a prediction engine.

Algorithm: flatten the Torah to one unbroken string of Hebrew consonants
(strip niqqud, cantillation marks, punctuation, and whitespace — the
Masoretic consonantal text is what ELS search operates on). For each
candidate skip distance d, scan every starting position i and check
whether the letters at i, i+d, i+2d, ... spell the target word.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DATA_FILE = Path(__file__).parent / "data" / "torah_full.json"

# Hebrew consonant block; niqqud/cantillation live outside this range and in
# the Unicode combining-mark categories, which unicodedata strips via NFKD.
_HEBREW_LETTER_RE = re.compile(r"[א-ת]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_to_consonants(verse_html: str) -> str:
    """Reduce a raw Sefaria verse string to bare Hebrew consonants."""
    no_tags = _HTML_TAG_RE.sub("", verse_html)
    # NFKD separates base letters from niqqud/cantillation combining marks
    decomposed = unicodedata.normalize("NFKD", no_tags)
    return "".join(_HEBREW_LETTER_RE.findall(decomposed))


@dataclass
class BookText:
    """A book's consonantal letter stream plus a position -> (chapter, verse) index."""

    title: str
    letters: str
    # positions[i] = (chapter_num, verse_num) that letter i belongs to (1-indexed)
    positions: list = field(default_factory=list)

    @classmethod
    def from_chapters(cls, title: str, chapters: list[list[str]]) -> "BookText":
        letters_parts = []
        positions = []
        for chapter_idx, chapter in enumerate(chapters, start=1):
            for verse_idx, verse in enumerate(chapter, start=1):
                consonants = strip_to_consonants(verse)
                letters_parts.append(consonants)
                positions.extend([(chapter_idx, verse_idx)] * len(consonants))
        return cls(title=title, letters="".join(letters_parts), positions=positions)


def load_torah(data_file: Path = DATA_FILE) -> dict[str, BookText]:
    """Load the full Torah and build per-book consonantal letter streams.

    Run fetch_full_torah.py first to produce data_file if it doesn't exist.
    """
    if not data_file.exists():
        raise FileNotFoundError(
            f"{data_file} not found — run `python3 bible_codes/fetch_full_torah.py` first"
        )
    with open(data_file, encoding="utf-8") as f:
        raw = json.load(f)
    return {title: BookText.from_chapters(title, chapters) for title, chapters in raw.items()}


@dataclass
class ELSMatch:
    book: str
    word: str
    skip: int
    start_index: int
    start_ref: tuple  # (chapter, verse) of the first letter
    end_ref: tuple  # (chapter, verse) of the last letter


def find_els(book: BookText, word: str, min_skip: int, max_skip: int) -> Iterator[ELSMatch]:
    """Search `book` for `word` at every skip distance in [min_skip, max_skip].

    Positive skip reads left-to-right through the letter stream; negative
    skip (min_skip may be negative) reads right-to-left, matching how ELS
    search is conventionally run in both directions over Hebrew text.
    """
    word = strip_to_consonants(word)
    if not word:
        return
    n = len(book.letters)
    word_len = len(word)

    for skip in range(min_skip, max_skip + 1):
        if skip == 0:
            continue
        step = abs(skip)
        span = step * (word_len - 1)
        if span >= n:
            continue
        for start in range(n - span):
            positions = range(start, start + span + 1, step) if skip > 0 else range(
                start + span, start - 1, -step
            )
            if all(book.letters[p] == word[k] for k, p in enumerate(positions)):
                idx_list = list(positions)
                yield ELSMatch(
                    book=book.title,
                    word=word,
                    skip=skip,
                    start_index=idx_list[0],
                    start_ref=book.positions[idx_list[0]],
                    end_ref=book.positions[idx_list[-1]],
                )


def search_all_books(
    books: dict[str, BookText], word: str, min_skip: int = -1000, max_skip: int = 1000
) -> list[ELSMatch]:
    matches = []
    for book in books.values():
        matches.extend(find_els(book, word, min_skip, max_skip))
    return matches


def render_matrix(
    book: BookText,
    match: ELSMatch,
    width: int | None = None,
    row_margin: int = 1,
    col_margin: int = 3,
) -> str:
    """Render a 2D cylinder letter grid around an ELS match.

    When width is omitted, defaults to abs(match.skip) (or 20 if |skip| == 1),
    which vertically aligns the letters of the matched word into a single column.
    """
    if width is None:
        width = abs(match.skip) if abs(match.skip) > 1 else 20
    W = width
    target_indices = set(match.start_index + i * match.skip for i in range(len(match.word)))
    min_idx = min(target_indices)
    max_idx = max(target_indices)
    min_row = min_idx // W
    max_row = max_idx // W
    target_col = match.start_index % W

    r_start = max(0, min_row - row_margin)
    r_end = min((len(book.letters) - 1) // W, max_row + row_margin)
    c_start = max(0, target_col - col_margin)
    c_end = min(W - 1, target_col + col_margin)

    lines = [
        f"2D Matrix Grid (Width={W}, Rows {r_start}..{r_end}, Cols {c_start}..{c_end}):"
    ]
    for r in range(r_start, r_end + 1):
        row_chars = []
        for c in range(c_start, c_end + 1):
            idx = r * W + c
            if idx < len(book.letters):
                ch = book.letters[idx]
                if idx in target_indices:
                    row_chars.append(f"[{ch}]")
                else:
                    row_chars.append(f" {ch} ")
            else:
                row_chars.append("   ")
        ref_idx = min(r * W + c_start, len(book.letters) - 1)
        ref = book.positions[ref_idx] if book.positions else (0, 0)
        lines.append(f"Row {r:3d} ({ref[0]}:{ref[1]:2d}) | " + "".join(row_chars))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word", help="Hebrew word to search for (consonants only, e.g. תורה)")
    parser.add_argument("--min-skip", type=int, default=-1000)
    parser.add_argument("--max-skip", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=20, help="max matches to print")
    parser.add_argument("--matrix", action="store_true", help="render 2D letter grid for first match")
    args = parser.parse_args()

    print(
        "NOTE: ELS 'Bible code' matches are a statistical artifact of searching "
        "long texts with many free parameters (skip, direction, spelling) — see "
        "McKay et al. 1999. This tool is for exploration/entertainment only.\n"
    )

    torah = load_torah()
    results = search_all_books(torah, args.word, args.min_skip, args.max_skip)
    print(f"Found {len(results)} match(es) for '{args.word}' across skips "
          f"[{args.min_skip}, {args.max_skip}]\n")
    for m in results[: args.limit]:
        print(
            f"  {m.book} skip={m.skip:+d}  "
            f"{m.book} {m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]}"
        )

    if args.matrix and results:
        top_match = results[0]
        book = torah[top_match.book]
        print("\n" + render_matrix(book, top_match) + "\n")
