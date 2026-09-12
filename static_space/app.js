// Client-side port of bible_codes/els_search.py and bible_codes/gematria.py.
//
// DISCLAIMER (same as the Python modules): ELS "Bible code" matches and
// gematria "hidden code" searches are exploratory pattern-matching, not
// validated discoveries. McKay, Bar-Natan, Bar-Hillel & Kalai (Statistical
// Science, 1999) showed the same methodology finds equally "significant"
// matches in non-religious texts like Moby Dick. This is a text-search toy.

const HEBREW_LETTER_RE = /[א-ת]/g;
const HTML_TAG_RE = /<[^>]+>/g;

const GEMATRIA_VALUES = {
  "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
  "י": 10, "כ": 20, "ל": 30, "מ": 40, "נ": 50, "ס": 60, "ע": 70, "פ": 80, "צ": 90,
  "ק": 100, "ר": 200, "ש": 300, "ת": 400,
};
// Final-form letters normalize to their regular value (same convention as gematria.py).
const FINAL_TO_REGULAR = { "ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ" };

function stripToConsonants(verseHtml) {
  const noTags = verseHtml.replace(HTML_TAG_RE, "");
  // NFKD separates base letters from niqqud/cantillation combining marks,
  // same as Python's unicodedata.normalize("NFKD", ...).
  const decomposed = noTags.normalize("NFKD");
  return (decomposed.match(HEBREW_LETTER_RE) || []).join("");
}

function letterValue(letter) {
  const base = FINAL_TO_REGULAR[letter] || letter;
  return GEMATRIA_VALUES[base];
}

function wordValue(word) {
  const consonants = stripToConsonants(word);
  let sum = 0;
  for (const c of consonants) sum += letterValue(c);
  return sum;
}

// Build { title -> { letters, positions, values, verseSums } } for every book.
function buildBooks(raw) {
  const books = {};
  for (const [title, chapters] of Object.entries(raw)) {
    let letters = "";
    const positions = []; // positions[i] = [chapter, verse]
    const verseSums = []; // [chapter, verse, sum]
    chapters.forEach((chapter, chapterIdx0) => {
      const chapterIdx = chapterIdx0 + 1;
      chapter.forEach((verse, verseIdx0) => {
        const verseIdx = verseIdx0 + 1;
        const consonants = stripToConsonants(verse);
        let verseSum = 0;
        for (const c of consonants) {
          positions.push([chapterIdx, verseIdx]);
          verseSum += letterValue(c);
        }
        letters += consonants;
        verseSums.push([chapterIdx, verseIdx, verseSum]);
      });
    });
    books[title] = { title, letters, positions, verseSums };
  }
  return books;
}

// ELS search: find every occurrence of `word` at skip distances in [minSkip, maxSkip].
function findEls(book, word, minSkip, maxSkip) {
  const target = stripToConsonants(word);
  if (!target) return [];
  const n = book.letters.length;
  const wordLen = target.length;
  const matches = [];

  for (let skip = minSkip; skip <= maxSkip; skip++) {
    if (skip === 0) continue;
    const step = Math.abs(skip);
    const span = step * (wordLen - 1);
    if (span >= n) continue;
    for (let start = 0; start < n - span; start++) {
      let ok = true;
      const idxs = new Array(wordLen);
      for (let k = 0; k < wordLen; k++) {
        const idx = skip > 0 ? start + k * step : start + span - k * step;
        if (book.letters[idx] !== target[k]) { ok = false; break; }
        idxs[k] = idx;
      }
      if (ok) {
        matches.push({
          book: book.title,
          word: target,
          skip,
          startIndex: idxs[0],
          startRef: book.positions[idxs[0]],
          endRef: book.positions[idxs[idxs.length - 1]],
        });
      }
    }
  }
  return matches;
}

// 2D cylinder grid render, mirroring els_search.py's render_matrix().
function renderMatrix(book, match, width, rowMargin = 1, colMargin = 3) {
  const W = width || (Math.abs(match.skip) > 1 ? Math.abs(match.skip) : 20);
  const wordLen = match.word.length;
  const targetIndices = new Set();
  for (let i = 0; i < wordLen; i++) targetIndices.add(match.startIndex + i * match.skip);

  const minIdx = Math.min(...targetIndices);
  const maxIdx = Math.max(...targetIndices);
  const minRow = Math.floor(minIdx / W);
  const maxRow = Math.floor(maxIdx / W);
  const targetCol = ((match.startIndex % W) + W) % W;

  const rStart = Math.max(0, minRow - rowMargin);
  const rEnd = Math.min(Math.floor((book.letters.length - 1) / W), maxRow + rowMargin);
  const cStart = Math.max(0, targetCol - colMargin);
  const cEnd = Math.min(W - 1, targetCol + colMargin);

  const lines = [`2D Matrix Grid (Width=${W}, Rows ${rStart}..${rEnd}, Cols ${cStart}..${cEnd}):`];
  for (let r = rStart; r <= rEnd; r++) {
    const rowChars = [];
    for (let c = cStart; c <= cEnd; c++) {
      const idx = r * W + c;
      if (idx < book.letters.length) {
        const ch = book.letters[idx];
        rowChars.push(targetIndices.has(idx) ? `[${ch}]` : ` ${ch} `);
      } else {
        rowChars.push("   ");
      }
    }
    const refIdx = Math.min(r * W + cStart, book.letters.length - 1);
    const ref = book.positions[refIdx] || [0, 0];
    lines.push(`Row ${String(r).padStart(3)} (${ref[0]}:${String(ref[1]).padStart(2)}) | ` + rowChars.join(""));
  }
  return lines.join("\n");
}

function findVersesWithValue(book, target) {
  return book.verseSums.filter(([, , total]) => total === target);
}

// Numeric analogue of ELS: equidistant runs of `length` letters summing to `target`.
function findSkipSum(book, target, length, minSkip, maxSkip) {
  const n = book.letters.length;
  const matches = [];
  const values = [];
  for (const c of book.letters) values.push(letterValue(c));

  for (let skip = minSkip; skip <= maxSkip; skip++) {
    if (skip === 0) continue;
    const step = Math.abs(skip);
    const span = step * (length - 1);
    if (span >= n) continue;
    for (let start = 0; start < n - span; start++) {
      let total = 0;
      const idxs = new Array(length);
      for (let k = 0; k < length; k++) {
        const idx = skip > 0 ? start + k * step : start + span - k * step;
        idxs[k] = idx;
        total += values[idx];
      }
      if (total === target) {
        matches.push({
          book: book.title, skip, length, total,
          startIndex: idxs[0],
          startRef: book.positions[idxs[0]],
          endRef: book.positions[idxs[idxs.length - 1]],
        });
      }
    }
  }
  return matches;
}

window.BibleCodes = {
  buildBooks, findEls, renderMatrix, wordValue, findVersesWithValue, findSkipSum, stripToConsonants,
};
