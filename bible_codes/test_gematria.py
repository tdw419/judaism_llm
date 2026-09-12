"""Correctness tests for gematria.py — deterministic, no network required."""

from els_search import BookText
from gematria import (
    NumericBook,
    find_skip_sum,
    find_verses_with_value,
    letter_value,
    to_values,
    word_value,
)


def test_letter_value_basic():
    assert letter_value("א") == 1
    assert letter_value("י") == 10
    assert letter_value("כ") == 20
    assert letter_value("ק") == 100
    assert letter_value("ת") == 400


def test_letter_value_final_forms_normalize_by_default():
    # ך normally counts as כ (=20), not the sofit value 500
    assert letter_value("ך") == 20
    assert letter_value("ך", sofit=True) == 500


def test_word_value_known_word():
    # תורה = ת(400) + ו(6) + ר(200) + ה(5) = 611
    assert word_value("תורה") == 611


def test_word_value_ignores_niqqud_and_tags():
    assert word_value("<big>בְּ</big>רֵאשִׁ֖ית") == word_value("בראשית")


def test_to_values_matches_letter_value():
    assert to_values("אבג") == [1, 2, 3]


def test_numeric_book_verse_sums():
    book = BookText.from_chapters("Test", [["אב", "גד"]])  # verse1=א(1)+ב(2)=3, verse2=ג(3)+ד(4)=7
    nbook = NumericBook.from_book_text(book)
    assert nbook.verse_sums == [(1, 1, 3), (1, 2, 7)]
    assert find_verses_with_value(nbook, 3) == [(1, 1, 3)]
    assert find_verses_with_value(nbook, 999) == []


def test_find_skip_sum():
    # letters: א(1) ב(2) ג(3) ד(4) ה(5)
    book = BookText.from_chapters("Test", [["אבגדה"]])
    nbook = NumericBook.from_book_text(book)
    # skip=1, length=2: consecutive pairs summing to 5 -> (1,4)? values are (1,2),(2,3),(3,4),(4,5)
    # sums: 3,5,7,9 -> target 5 matches index (1,2)=values(2,3)
    matches = list(find_skip_sum(nbook, target=5, length=2, min_skip=1, max_skip=1))
    assert len(matches) == 1
    assert matches[0].start_index == 1

    # skip=2, length=2: pairs (0,2)=1+3=4, (1,3)=2+4=6, (2,4)=3+5=8
    matches = list(find_skip_sum(nbook, target=6, length=2, min_skip=2, max_skip=2))
    assert len(matches) == 1
    assert matches[0].start_index == 1


if __name__ == "__main__":
    import sys

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failures else 0)
