"""Unicode-aware text normalization shared across modules.

These helpers exist because the obvious ASCII versions silently break every
non-Latin script. ``re.sub(r"[^a-z0-9]+", " ", s.lower())`` reduces "как
экономить деньги" to an empty string, which meant Russian, Arabic, Greek, Hindi
and CJK topics were dropped by the trend aggregator and never matched by
duplicate detection — with no error to explain why.

Character classes are decided by Unicode *category* rather than by ``\\w``,
because ``\\w`` excludes combining marks: it mangles "कैसे" into "क स", corrupting
Devanagari, Thai, Arabic vowel marks and any decomposed accent that survives
normalization. Categories starting with L (letter), N (number) or M (mark) cover
all of those.

``casefold`` is used instead of ``lower`` because it folds cases ``lower`` does
not, such as German ß to ss.
"""

from __future__ import annotations

import unicodedata

# Apostrophes are word-internal in many languages ("don't", "l'argent"), so they
# are kept when they sit between other word characters.
_APOSTROPHES = frozenset("'’ʼ")


def _is_word_char(char: str) -> bool:
    return unicodedata.category(char)[0] in ("L", "N", "M")


def normalize_key(text: str) -> str:
    """Collapse text to a comparison key, ignoring case, punctuation and spacing.

    NFKC normalization is applied first so visually identical strings that differ
    in codepoints (full-width Latin, composed vs decomposed accents) compare
    equal. Runs of non-word characters collapse to a single space.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()

    out: list[str] = []
    pending_space = False
    for char in folded:
        if _is_word_char(char):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append(char)
        else:
            pending_space = True

    return "".join(out)


def word_set(text: str, *, min_length: int = 3, stopwords: set[str] | None = None) -> set[str]:
    """Extract distinct words for relevance comparison, in any script.

    ``min_length`` filters noise words. Note it is a blunt instrument for CJK,
    where a two-character token is often a whole concept, so CJK niches may match
    less precisely than alphabetic ones.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()

    words: set[str] = set()
    current: list[str] = []

    def flush() -> None:
        token = "".join(current).strip("".join(_APOSTROPHES))
        if len(token) >= min_length:
            words.add(token)
        current.clear()

    for char in folded:
        if _is_word_char(char) or (char in _APOSTROPHES and current):
            current.append(char)
        else:
            flush()
    flush()

    if stopwords:
        words -= stopwords
    return words
