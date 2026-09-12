#!/usr/bin/env python3
"""Fetch the complete Hebrew text of the Torah (all chapters/verses) from Sefaria.

The texts already downloaded under sefaria_texts/Tanakh_Torah only contain
chapter 1 of each book (download_sefaria.py requests each title with no
section range, which Sefaria truncates to the default section). ELS search
needs the unbroken full-book letter stream, so this fetches each book as a
single chapter range (e.g. "Genesis.1-50").
"""

import json
import time
from pathlib import Path

import requests

SEFARIA_API_BASE = "https://www.sefaria.org/api"

# (book title, chapter count)
TORAH_BOOKS = [
    ("Genesis", 50),
    ("Exodus", 40),
    ("Leviticus", 27),
    ("Numbers", 36),
    ("Deuteronomy", 34),
]

OUTPUT_DIR = Path(__file__).parent / "data"


def fetch_book(title: str, num_chapters: int) -> dict:
    url = f"{SEFARIA_API_BASE}/texts/{title}.1-{num_chapters}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_torah(output_dir: Path = OUTPUT_DIR, delay: float = 0.3) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "torah_full.json"

    books = {}
    for title, num_chapters in TORAH_BOOKS:
        print(f"Fetching {title} (1-{num_chapters})...")
        data = fetch_book(title, num_chapters)
        he = data["he"]
        verse_count = sum(len(chapter) for chapter in he)
        print(f"  {len(he)} chapters, {verse_count} verses")
        books[title] = he
        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False)

    print(f"\nSaved full Torah text to {output_file}")
    return output_file


if __name__ == "__main__":
    fetch_all_torah()
