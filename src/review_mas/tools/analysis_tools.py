"""Tools for the Analysis agent: deterministic stats and keyword extraction.

Outputs are computed from review text and ratings only (no LLM). Optional Ollama
formatting happens in the agent node, not inside these tools.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from review_mas.state import ReviewRecord

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "my",
        "your",
        "their",
        "with",
        "from",
        "by",
        "not",
        "no",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "about",
        "into",
        "out",
        "up",
        "down",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "such",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of length >= 3."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in raw if len(t) >= 3 and t not in _STOPWORDS]


def compute_review_statistics(reviews: list[ReviewRecord]) -> dict[str, Any]:
    """Compute rating distribution and sentiment buckets from review records.

    Sentiment buckets are **rule-based** (not an ML model):
    - rating >= 4 → positive
    - rating <= 2 and > 0 → negative
    - rating == 3 → neutral
    - rating == 0 → unknown (missing rating)

    Args:
        reviews: Normalized review dicts with at least ``rating`` (int).

    Returns:
        JSON-serializable summary dict for downstream agents and reporting.
    """
    n = len(reviews)
    hist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    unknown = 0
    pos = neg = neu = 0
    sum_r = 0.0
    counted = 0

    for r in reviews:
        rating = int(r.get("rating") or 0)
        if rating <= 0:
            unknown += 1
            continue
        rating = max(1, min(5, rating))
        hist[rating] += 1
        sum_r += rating
        counted += 1
        if rating >= 4:
            pos += 1
        elif rating <= 2:
            neg += 1
        else:
            neu += 1

    avg = round(sum_r / counted, 2) if counted else 0.0
    return {
        "n_reviews": n,
        "n_rated": counted,
        "n_unknown_rating": unknown,
        "rating_histogram_1_to_5": hist,
        "avg_rating": avg,
        "sentiment_counts_rule_based": {
            "positive_rating_ge_4": pos,
            "negative_rating_le_2": neg,
            "neutral_rating_3": neu,
        },
    }


def extract_top_keywords(
    reviews: list[ReviewRecord],
    top_n: int = 20,
) -> list[tuple[str, int]]:
    """Extract most frequent content words across all review_text fields.

    Args:
        reviews: Review dicts with ``review_text``.
        top_n: Number of top terms to return.

    Returns:
        List of ``(term, count)`` sorted by count descending.
    """
    counter: Counter[str] = Counter()
    for r in reviews:
        text = r.get("review_text") or ""
        counter.update(_tokenize(text))
    return counter.most_common(top_n)


def sentiment_label_from_rating(rating: int) -> tuple[str, float]:
    """Map star rating to a coarse label and score for enrichment."""
    r = int(rating or 0)
    if r <= 0:
        return "unknown", 0.0
    if r >= 4:
        return "positive", 0.7
    if r <= 2:
        return "negative", -0.7
    return "neutral", 0.0


def topics_for_review(review_text: str, top_terms: list[str], max_topics: int = 5) -> list[str]:
    """Attach up to ``max_topics`` global keywords that appear in this review text."""
    if not top_terms:
        return []
    lower = (review_text or "").lower()
    out: list[str] = []
    for term in top_terms:
        if term in lower:
            out.append(term)
        if len(out) >= max_topics:
            break
    return out
