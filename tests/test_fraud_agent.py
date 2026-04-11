from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_duplicate_text_flags_second_review():
    from review_mas.tools.fraud_tools import assess_fraud_patterns

    reviews = [
        {
            "review_id": "1",
            "product_id": "p",
            "rating": 5,
            "review_text": "Great product same",
            "review_date": "",
            "sentiment_label": "positive",
            "sentiment_score": 0.7,
            "topics": [],
        },
        {
            "review_id": "2",
            "product_id": "p",
            "rating": 5,
            "review_text": "Great product same",
            "review_date": "",
            "sentiment_label": "positive",
            "sentiment_score": 0.7,
            "topics": [],
        },
    ]
    assessments, summary = assess_fraud_patterns(reviews)
    assert summary["flagged_count"] >= 1
    flagged_ids = {a["review_id"] for a in assessments if a.get("flagged")}
    assert "2" in flagged_ids


def test_high_rating_very_short_text_flags():
    from review_mas.tools.fraud_tools import assess_fraud_patterns

    reviews = [
        {
            "review_id": "1",
            "product_id": "p",
            "rating": 5,
            "review_text": "ok",
            "review_date": "",
            "sentiment_label": "positive",
            "sentiment_score": 0.7,
            "topics": [],
        },
    ]
    assessments, summary = assess_fraud_patterns(reviews)
    assert assessments[0].get("flagged") is True
    assert "high_rating_very_short_text" in (assessments[0].get("reasons") or [])


def test_normalize_strips_html():
    from review_mas.tools.fraud_tools import normalize_review_text

    s = normalize_review_text("Hello<br/>World")
    assert s == "hello world"
