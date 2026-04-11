from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_compute_review_statistics_histogram_and_sentiment_buckets():
    from review_mas.tools.analysis_tools import compute_review_statistics

    reviews = [
        {"review_id": "1", "product_id": "p", "rating": 5, "review_text": "a", "review_date": ""},
        {"review_id": "2", "product_id": "p", "rating": 4, "review_text": "b", "review_date": ""},
        {"review_id": "3", "product_id": "p", "rating": 3, "review_text": "c", "review_date": ""},
        {"review_id": "4", "product_id": "p", "rating": 2, "review_text": "d", "review_date": ""},
        {"review_id": "5", "product_id": "p", "rating": 0, "review_text": "e", "review_date": ""},
    ]
    s = compute_review_statistics(reviews)
    assert s["n_reviews"] == 5
    assert s["n_unknown_rating"] == 1
    assert s["n_rated"] == 4
    assert s["rating_histogram_1_to_5"][5] == 1
    assert s["rating_histogram_1_to_5"][4] == 1
    assert s["rating_histogram_1_to_5"][3] == 1
    assert s["rating_histogram_1_to_5"][2] == 1
    assert s["sentiment_counts_rule_based"]["positive_rating_ge_4"] == 2
    assert s["sentiment_counts_rule_based"]["neutral_rating_3"] == 1
    assert s["sentiment_counts_rule_based"]["negative_rating_le_2"] == 1
    assert s["avg_rating"] == 3.5


def test_extract_top_keywords_orders_by_frequency():
    from review_mas.tools.analysis_tools import extract_top_keywords

    reviews = [
        {
            "review_id": "1",
            "product_id": "p",
            "rating": 5,
            "review_text": "battery battery life excellent battery",
            "review_date": "",
        },
        {"review_id": "2", "product_id": "p", "rating": 4, "review_text": "screen good", "review_date": ""},
    ]
    top = extract_top_keywords(reviews, top_n=5)
    assert top[0][0] == "battery"
    assert top[0][1] >= 3


def test_topics_for_review_overlap():
    from review_mas.tools.analysis_tools import topics_for_review

    t = topics_for_review("The battery life is great", ["battery", "screen"], max_topics=3)
    assert t == ["battery"]
