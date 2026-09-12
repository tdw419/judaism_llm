#!/usr/bin/env python3
"""Gematria: map Torah letters to numbers (mispar hechrachi) for numeric analysis.

DISCLAIMER: same caveat as els_search.py. Gematria word-sums are a real,
centuries-old feature of Jewish textual tradition (used homiletically, e.g.
two words sharing a numeric value are linked in a drash), but "hidden codes"
claims built on top of it -- verse sums matching dates, running sums hitting
significant numbers, etc. -- are exactly the kind of after-the-fact pattern
search McKay et al. (1999) showed produces "hits" in any sufficiently long
text once you're free to pick the word, the skip, the target number, and the
matching rule after seeing the data. Use this for exploring the traditional
gematria system and running your own numeric experiments, not for treating
matches as validated discoveries.

Standard values (mispar hechrachi):
  א=1  ב=2  ג=3  ד=4  ה=5  ו=6  ז=7  ח=8  ט=9
  י=10 כ=20 ל=30 מ=40 נ=50 ס=60 ע=70 פ=80 צ=90
  ק=100 ר=200 ש=300 ת=400

Final-form letters (ך ם ן ף ץ) are normalized to their regular value, which
is the common convention (some traditions instead give them 500-900 --
see FINAL_FORM_VALUES below if you want that variant).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from els_search import BookText, load_torah, strip_to_consonants  # noqa: E402

GEMATRIA_VALUES = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ל": 30, "מ": 40, "נ": 50, "ס": 60, "ע": 70, "פ": 80, "צ": 90,
    "ק": 100, "ר": 200, "ש": 300, "ת": 400,
}

# Alternate ("sofit") convention some traditions use for final-letter forms.
FINAL_FORM_VALUES = {"ך": 500, "ם": 600, "ן": 700, "ף": 800, "ץ": 900}

# strip_to_consonants() normalizes final forms to their regular letter already
# (NFKD does not do this on its own -- final letters are distinct codepoints),
# so map them explicitly here for standalone use of this module.
_FINAL_TO_REGULAR = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}


def letter_value(letter: str, sofit: bool = False) -> int:
    """Numeric value of a single Hebrew letter."""
    if sofit and letter in FINAL_FORM_VALUES:
        return FINAL_FORM_VALUES[letter]
    base = _FINAL_TO_REGULAR.get(letter, letter)
    return GEMATRIA_VALUES[base]


def to_values(letters: str, sofit: bool = False) -> list[int]:
    """Convert a consonant string to its list of gematria values."""
    return [letter_value(c, sofit=sofit) for c in letters]


def word_value(word: str, sofit: bool = False) -> int:
    """Sum the gematria value of a word (spaces/punctuation ignored)."""
    return sum(to_values(strip_to_consonants(word), sofit=sofit))


@dataclass
class NumericBook:
    """A book's letter values plus per-verse (chapter, verse, value) sums."""

    title: str
    values: list  # values[i] is the gematria value of letters[i]
    positions: list  # positions[i] = (chapter, verse) that letter i belongs to
    verse_sums: list  # list of (chapter, verse, sum_of_letter_values)

    @classmethod
    def from_book_text(cls, book: BookText, sofit: bool = False) -> "NumericBook":
        values = to_values(book.letters, sofit=sofit)
        verse_totals = {}
        verse_order = []
        for pos, val in zip(book.positions, values):
            if pos not in verse_totals:
                verse_totals[pos] = 0
                verse_order.append(pos)
            verse_totals[pos] += val
        verse_sums = [(ch, vs, verse_totals[(ch, vs)]) for ch, vs in verse_order]
        return cls(title=book.title, values=values, positions=book.positions, verse_sums=verse_sums)


def load_numeric_torah(sofit: bool = False, data_file: Path = None) -> dict:
    """Load the Torah and convert every book to gematria values."""
    kwargs = {"data_file": data_file} if data_file else {}
    torah = load_torah(**kwargs)
    return {title: NumericBook.from_book_text(book, sofit=sofit) for title, book in torah.items()}


@dataclass
class SkipSumMatch:
    book: str
    skip: int
    start_index: int
    length: int
    total: int
    start_ref: tuple
    end_ref: tuple


def find_skip_sum(
    nbook: NumericBook, target: int, length: int, min_skip: int, max_skip: int
) -> Iterator[SkipSumMatch]:
    """Find every equidistant run of `length` letters whose values sum to `target`.

    This is the numeric analogue of ELS: instead of matching letters to a
    fixed word, it checks whether the values at positions i, i+skip,
    i+2*skip, ... (for `length` letters) add up to `target`.
    """
    n = len(nbook.values)
    for skip in range(min_skip, max_skip + 1):
        if skip == 0:
            continue
        step = abs(skip)
        span = step * (length - 1)
        if span >= n:
            continue
        for start in range(n - span):
            idxs = range(start, start + span + 1, step) if skip > 0 else range(
                start + span, start - 1, -step
            )
            idxs = list(idxs)
            total = sum(nbook.values[i] for i in idxs)
            if total == target:
                yield SkipSumMatch(
                    book=nbook.title,
                    skip=skip,
                    start_index=idxs[0],
                    length=length,
                    total=total,
                    start_ref=nbook.positions[idxs[0]],
                    end_ref=nbook.positions[idxs[-1]],
                )


def find_verses_with_value(nbook: NumericBook, target: int) -> list:
    """All verses whose full letter-value sum equals `target`."""
    return [(ch, vs, total) for ch, vs, total in nbook.verse_sums if total == target]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_word = sub.add_parser("word", help="compute the gematria value of a word")
    p_word.add_argument("word")

    p_verse = sub.add_parser("verse-sum", help="find verses whose total value equals a target")
    p_verse.add_argument("target", type=int)
    p_verse.add_argument("--limit", type=int, default=20)

    p_skip = sub.add_parser("skip-sum", help="find equidistant letter runs summing to a target")
    p_skip.add_argument("target", type=int)
    p_skip.add_argument("--length", type=int, default=3, help="number of letters in the run")
    p_skip.add_argument("--min-skip", type=int, default=-50)
    p_skip.add_argument("--max-skip", type=int, default=50)
    p_skip.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    print(
        "NOTE: gematria 'hidden code' searches are exploratory pattern-matching, "
        "not validated discoveries -- see this file's module docstring.\n"
    )

    if args.cmd == "word":
        print(f"{args.word} = {word_value(args.word)}")
    else:
        torah = load_numeric_torah()
        if args.cmd == "verse-sum":
            for title, nbook in torah.items():
                hits = find_verses_with_value(nbook, args.target)
                for ch, vs, total in hits[: args.limit]:
                    print(f"  {title} {ch}:{vs} = {total}")
        elif args.cmd == "skip-sum":
            for title, nbook in torah.items():
                count = 0
                for m in find_skip_sum(nbook, args.target, args.length, args.min_skip, args.max_skip):
                    print(f"  {m.book} skip={m.skip:+d} len={m.length} "
                          f"{m.start_ref[0]}:{m.start_ref[1]} -> {m.end_ref[0]}:{m.end_ref[1]} = {m.total}")
                    count += 1
                    if count >= args.limit:
                        break
