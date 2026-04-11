"""Tools for the Fraud Detection agent (rule + statistics based, no cloud).

These functions produce structured per-review assessments and an aggregate
``fraud_summary`` for observability and for optional local LLM explanation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from review_mas.state import EnrichedReview, FraudAssessment


def normalize_review_text(text: str) -> str:
    """Normalize review text for duplicate and length heuristics.

    - Lowercase
    - Strip HTML-like tags
    - Collapse whitespace

    Args:
        text: Raw review text.

    Returns:
        Normalized single-line string.
    """
    t = text or ""
    t = re.sub(r"<[^>]+>", " ", t, flags=re.IGNORECASE)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _score_from_reasons(reasons: list[str]) -> float:
    """Map reason codes to a bounded fraud score in [0, 1]."""
    weights = {
        "duplicate_review_text": 0.82,
        "very_short_review": 0.28,
        "high_rating_very_short_text": 0.35,
        "low_rating_very_short_text": 0.18,
        "mostly_non_letters": 0.22,
    }
    s = 0.05
    for r in reasons:
        s += weights.get(r, 0.1)
    return float(min(0.95, s))


def assess_fraud_patterns(
    reviews: list[EnrichedReview],
) -> tuple[list[FraudAssessment], dict[str, Any]]:
    """Assess simple fraud / spam signals for each enriched review.

    Signals (non-exhaustive, explainable):
    - **duplicate_review_text**: same normalized text seen more than once
    - **very_short_review**: normalized length is very small
    - **high_rating_very_short_text**: 5 stars but very little text
    - **low_rating_very_short_text**: 1 star but very little text
    - **mostly_non_letters**: text is mostly symbols/numbers/punctuation

    A review is **flagged** when the computed score is >= 0.45 or duplicate text is detected.

    Args:
        reviews: Output of the Analysis agent (must include ``review_id``, ``rating``, ``review_text``).

    Returns:
        Tuple of (per-review assessments, aggregate ``fraud_summary`` dict).
    """
    seen_norm: dict[str, str] = {}
    assessments: list[FraudAssessment] = []
    reason_counter: Counter[str] = Counter()

    for r in reviews:
        rid = str(r.get("review_id", "")).strip()
        rating = int(r.get("rating") or 0)
        raw_text = r.get("review_text") or ""
        norm = normalize_review_text(raw_text)

        reasons: list[str] = []

        if norm:
            first_id = seen_norm.get(norm)
            if first_id is not None and first_id != rid:
                reasons.append("duplicate_review_text")
            else:
                seen_norm.setdefault(norm, rid)

        if len(norm) < 18:
            reasons.append("very_short_review")

        if rating >= 5 and len(norm) < 45:
            reasons.append("high_rating_very_short_text")
        if rating == 1 and 0 < len(norm) < 35:
            reasons.append("low_rating_very_short_text")

        letters = sum(1 for ch in norm if ch.isalpha())
        if len(norm) >= 12 and letters / max(len(norm), 1) < 0.35:
            reasons.append("mostly_non_letters")

        # de-dupe reasons while preserving order
        reasons = list(dict.fromkeys(reasons))

        score = _score_from_reasons(reasons)
        flagged = bool(score >= 0.45 or "duplicate_review_text" in reasons)

        for reason in reasons:
            reason_counter[reason] += 1

        assessments.append(
            {
                "review_id": rid,
                "fraud_score": round(score, 3),
                "flagged": flagged,
                "reasons": reasons,
            }
        )

    flagged_n = sum(1 for a in assessments if a.get("flagged"))
    summary: dict[str, Any] = {
        "n_reviews": len(reviews),
        "flagged_count": flagged_n,
        "reason_counts": dict(reason_counter),
        "avg_fraud_score": round(
            sum(float(a.get("fraud_score") or 0) for a in assessments) / max(len(assessments), 1),
            3,
        ),
    }
    return assessments, summary
