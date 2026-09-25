"""Scoring for the multilingual regression suite (pure functions, unit-tested).

* ``cer`` - character error rate of a transcript against the source text,
  computed on letters/digits/combining marks only (case, spaces and
  punctuation ignored), so it works the same for spaced and unspaced scripts.
* ``content_recall`` - share of the English reference's content words that
  appear in the tool's English translation (light stemming, stopwords
  removed). A translation is free to rephrase, so this measures whether the
  meaning came through rather than exact wording.
* ``detected_language`` - reads the language code from the HTML dashboard,
  which both the Python and PowerShell tools render identically.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

STOPWORDS = frozenset("""
a an the and or but if so of to in on at by for from with without into onto
about as is are was were be been being am do does did done have has had having
it its it's this that these those there here then than too very can could will
would shall should may might must not no nor only also just our your their his
her my we you they he she i me him them us who whom which what when where why
how all any each every some such more most many much few other another own same
up down out over under again once let lets
""".split())

_WORD = re.compile(r"[a-z]+")


def normalize_chars(text: str) -> str:
    """NFKC, lowercase, keep only letters, digits and combining marks."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LNM")


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions, deletions, substitutions), O(len(a)*len(b))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference`` (0 = perfect)."""
    ref = normalize_chars(reference)
    hyp = normalize_chars(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set:
    """Stemmed English content words (stopwords and 1-2 letter words dropped)."""
    words = _WORD.findall((text or "").lower().replace("’", "'"))
    return {_stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def content_recall(reference_en: str, translation_en: str) -> float:
    """Fraction of the reference's content words found in the translation."""
    ref = content_words(reference_en)
    if not ref:
        return 1.0
    return len(ref & content_words(translation_en)) / len(ref)


_LANG = re.compile(r"Language detected</p>\s*<div class=\"value\">[^<]*?\(([A-Za-z-]+)\)</div>")


def detected_language(dashboard_html: str) -> Optional[str]:
    """The ``(code)`` shown in the dashboard's "Language detected" stat."""
    m = _LANG.search(dashboard_html or "")
    return m.group(1).lower() if m else None
