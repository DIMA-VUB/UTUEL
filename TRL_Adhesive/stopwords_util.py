"""
stopwords_util.py
Lightweight, dependency-free English stop-word removal used to build the
"stop-word" (SW) variant of each question.

The SW variant is the original question with common English function words
removed, keeping only the more content-bearing tokens.  Punctuation attached to
kept tokens is preserved so ``"What is the dielectric strength?"`` becomes
``"dielectric strength?"``.
"""

from __future__ import annotations

import string

# Curated English stop-word list (superset of the common NLTK / sklearn cores).
STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
    "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
})

_PUNCT = string.punctuation


def remove_stopwords(text: str) -> str:
    """Return *text* with English stop-words removed.

    Tokens are split on whitespace (so multi-part tokens such as
    ``"50Hz/1kHz/10kHz?"`` stay intact); a token is dropped when its
    lower-cased, punctuation-stripped core is a stop-word.  If removing
    stop-words would leave an empty string, the original *text* is returned so
    the SW variant is never empty.
    """
    if not text:
        return text

    kept = [
        tok for tok in text.split()
        if tok.strip(_PUNCT).lower() not in STOP_WORDS
    ]
    return " ".join(kept).strip() or text
