"""Rule-based Tamil -> Latin transliteration.

Two consumers:

* the corpus builder (``build_codeswitch_corpus.py``) uses it to synthesise the
  romanized/Tanglish register without an external transliteration model;
* the evaluation suite (``run_full_evaluation.py``) uses it to score CER after
  putting reference and hypothesis in the *same* alphabet - ASR models freely
  switch between Tamil script and Latin for code-mixed speech, which otherwise
  inflates character error rates for reasons unrelated to intelligibility.

The scheme is deliberately simple and deterministic: consonant + inherent vowel,
with the vowel signs / virama handled explicitly.  It is not a linguistically
complete transliteration engine, but it is stable, reviewable, and good enough
for both uses above (documented as such in the paper).
"""

from __future__ import annotations

import re
from typing import Dict, List

TAMIL_CONSONANTS: Dict[str, str] = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t", "ண": "n", "த": "th",
    "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "zh", "ள": "l", "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s",
    "ஹ": "h", "ஶ": "sh",
}

TAMIL_VOWELS: Dict[str, str] = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e",
    "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
}

VOWEL_SIGNS: Dict[str, str] = {
    "\u0bbe": "aa", "\u0bbf": "i", "\u0bc0": "ee", "\u0bc1": "u", "\u0bc2": "oo",
    "\u0bc6": "e", "\u0bc7": "e", "\u0bc8": "ai", "\u0bca": "o", "\u0bcb": "o",
    "\u0bcc": "au",
}

VIRAMA = "\u0bcd"
AYTHAM = "\u0b83"

TAMIL_RANGE = re.compile(r"[\u0b80-\u0bff]")

__all__ = [
    "TAMIL_CONSONANTS", "TAMIL_VOWELS", "VOWEL_SIGNS", "VIRAMA", "AYTHAM",
    "TAMIL_RANGE", "transliterate_word", "latinize_text", "has_tamil",
]


def has_tamil(text: str) -> bool:
    """True if ``text`` contains at least one Tamil-script character."""
    return bool(TAMIL_RANGE.search(text or ""))


def transliterate_word(word: str) -> str:
    """Best-effort Tamil -> Latin transliteration of a single token.

    Token-internal Latin/digits/punctuation are preserved, so mixed tokens
    (``order``, ``OTP1234``, ``கடை.``) behave sensibly.
    """
    if not has_tamil(word):
        return word
    out: List[str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in TAMIL_CONSONANTS:
            base = TAMIL_CONSONANTS[ch]
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in VOWEL_SIGNS:
                out.append(base + VOWEL_SIGNS[nxt])
                i += 2
                continue
            if nxt == VIRAMA:
                out.append(base)
                i += 2
                continue
            out.append(base + "a")  # inherent vowel
            i += 1
            continue
        if ch in TAMIL_VOWELS:
            out.append(TAMIL_VOWELS[ch])
            i += 1
            continue
        if ch == AYTHAM:
            out.append("h")
            i += 1
            continue
        if ch == VIRAMA:
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def latinize_text(text: str) -> str:
    """Transliterates every Tamil token in ``text`` to Latin, keeping the rest.

    Punctuation and Latin tokens pass through, so the result can be fed straight
    into CER/WER scoring.
    """
    if not has_tamil(text):
        return text
    return " ".join(transliterate_word(token) for token in str(text).split())
