"""Global state passed between agents (extend as the pipeline grows)."""

from __future__ import annotations

from typing import Any, TypedDict


class ReviewRecord(TypedDict, total=False):
    """Single review after scraping/normalization."""

    review_id: str
    product_id: str
    rating: int
    review_text: str
    review_date: str


class EnrichedReview(ReviewRecord, total=False):
    """Review + analysis fields (filled by Analysis agent)."""

    sentiment_label: str
    sentiment_score: float
    topics: list[str]


class FraudAssessment(TypedDict, total=False):
    """Output of Fraud Detection agent for one review."""

    review_id: str
    fraud_score: float
    flagged: bool
    reasons: list[str]


class GraphState(TypedDict, total=False):
    """LangGraph state: keys updated incrementally by each node."""

    dataset_path: str
    product_id: str
    auto_product: bool
    use_ollama: bool
    ollama_model: str
    raw_reviews: list[ReviewRecord]
    # Grounded facts from Analysis (JSON-serializable) for Fraud/Recommendation
    analysis_summary: dict[str, Any]
    enriched_reviews: list[EnrichedReview]
    fraud_assessments: list[FraudAssessment]
    # Aggregate fraud stats + optional grounded LLM explanation
    fraud_summary: dict[str, Any]
    final_report: str
    trace: list[dict[str, Any]]
