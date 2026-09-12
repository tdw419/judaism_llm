"""Correctness tests for els_search.py — deterministic, no network required."""

from els_search import BookText, find_els, strip_to_consonants


def test_strip_to_consonants_removes_niqqud_and_tags():
    raw = "<big>בְּ</big>רֵאשִׁ֖ית"
    assert strip_to_consonants(raw) == "בראשית"


def test_find_els_skip_1_is_plain_substring():
    book = BookText.from_chapters("Test", [["אבגדהוזחטי"]])
    matches = list(find_els(book, "גדה", min_skip=1, max_skip=1))
    assert len(matches) == 1
    assert matches[0].start_index == 2

    matches = list(find_els(book, "זזז", min_skip=1, max_skip=1))
    assert matches == []


def test_find_els_positive_skip():
    # letters:  א  ב  ג  ד  ה  ו  ז  ח  ט  י
    # index:    0  1  2  3  4  5  6  7  8  9
    # skip=3 from index 0: א(0) ד(3) ז(6)  -> "אדז"
    book = BookText.from_chapters("Test", [["אבגדהוזחטי"]])
    matches = list(find_els(book, "אדז", min_skip=3, max_skip=3))
    assert len(matches) == 1
    assert matches[0].start_index == 0
    assert matches[0].skip == 3


def test_find_els_negative_skip_reads_backwards():
    # skip=-3 should read the mirror direction of skip=3
    book = BookText.from_chapters("Test", [["אבגדהוזחטי"]])
    forward = list(find_els(book, "אדז", min_skip=3, max_skip=3))
    backward = list(find_els(book, "זדא", min_skip=-3, max_skip=-3))
    assert len(forward) == 1 and len(backward) == 1
    assert forward[0].start_ref == backward[0].end_ref


def test_position_index_maps_to_correct_verse():
    book = BookText.from_chapters("Test", [["אבג", "דהו"], ["זחט"]])
    assert book.letters == "אבגדהוזחט"
    assert book.positions[0] == (1, 1)
    assert book.positions[3] == (1, 2)
    assert book.positions[6] == (2, 1)


def test_render_matrix_highlights_letters():
    from els_search import render_matrix
    # letters:  א  ב  ג  ד  ה  ו  ז  ח  ט
    # row 0:    א  ב  ג
    # row 1:    ד  ה  ו
    # row 2:    ז  ח  ט
    # skip=3 from index 1 (ב) gives ב(1), ה(4), ח(7) -> "בהח"
    book = BookText.from_chapters("Test", [["אבגדהוזחט"]])
    matches = list(find_els(book, "בהח", min_skip=3, max_skip=3))
    assert len(matches) == 1
    grid = render_matrix(book, matches[0], width=3, row_margin=0, col_margin=1)
    assert "[ב]" in grid
    assert "[ה]" in grid
    assert "[ח]" in grid


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
