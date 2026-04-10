"""Agent steps (stubs). Replace with real tools + Ollama reasoning per teammate."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from review_mas.state import EnrichedReview, FraudAssessment, GraphState, ReviewRecord

logger = logging.getLogger(__name__)


def _append_trace(state: GraphState, agent: str, detail: str) -> list[dict]:
    trace = list(state.get("trace") or [])
    trace.append({"agent": agent, "detail": detail})
    return trace


def scraper_node(state: GraphState) -> dict:
    """Agent 1: load raw reviews from local CSV (replace with dedicated tools later)."""
    path = Path(state.get("dataset_path") or "data/sample_reviews.csv")
    if not path.is_file():
        logger.error("Dataset not found: %s", path)
        return {
            "raw_reviews": [],
            "trace": _append_trace(state, "scraper", f"missing file: {path}"),
        }

    rows: list[ReviewRecord] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "product_id": row.get("product_id", ""),
                    "rating": int(row["rating"]) if row.get("rating") else 0,
                    "review_text": (row.get("review_text") or "").strip(),
                    "review_date": row.get("review_date") or "",
                }
            )

    logger.info("Scraper loaded %d reviews from %s", len(rows), path)
    return {
        "raw_reviews": rows,
        "trace": _append_trace(state, "scraper", f"loaded {len(rows)} rows"),
    }


def analysis_node(state: GraphState) -> dict:
    """Agent 2: sentiment/topics (stub — add model + keyword tools)."""
    enriched: list[EnrichedReview] = []
    for r in state.get("raw_reviews") or []:
        text = r.get("review_text") or ""
        label = "positive" if r.get("rating", 0) >= 4 else "negative"
        enriched.append(
            {
                **r,
                "sentiment_label": label,
                "sentiment_score": 0.5 if label == "positive" else -0.5,
                "topics": [],
            }
        )
    logger.info("Analysis enriched %d reviews (stub)", len(enriched))
    return {
        "enriched_reviews": enriched,
        "trace": _append_trace(state, "analysis", "stub sentiment by rating"),
    }


def fraud_node(state: GraphState) -> dict:
    """Agent 3: fraud signals (stub — duplicate text detection)."""
    seen_text: dict[str, str] = {}
    assessments: list[FraudAssessment] = []
    for r in state.get("enriched_reviews") or []:
        rid = r.get("review_id", "")
        text = (r.get("review_text") or "").strip().lower()
        dup_of = seen_text.get(text)
        flagged = dup_of is not None
        if not flagged:
            seen_text[text] = rid
        reasons = ["duplicate_review_text"] if flagged else []
        score = 0.85 if flagged else 0.1
        assessments.append(
            {
                "review_id": rid,
                "fraud_score": score,
                "flagged": flagged,
                "reasons": reasons,
            }
        )
    flagged_n = sum(1 for a in assessments if a.get("flagged"))
    logger.info("Fraud assessed %d reviews, %d flagged (stub)", len(assessments), flagged_n)
    return {
        "fraud_assessments": assessments,
        "trace": _append_trace(
            state, "fraud", f"stub duplicate check; flagged={flagged_n}"
        ),
    }


def recommendation_node(state: GraphState) -> dict:
    """Agent 4: buyer report (stub — replace with Ollama + report tool)."""
    n = len(state.get("enriched_reviews") or [])
    flags = sum(1 for a in state.get("fraud_assessments") or [] if a.get("flagged"))
    report = (
        f"## Should you buy? (stub)\n\n"
        f"- Reviews analyzed: **{n}**\n"
        f"- Suspicious / duplicate-text flags: **{flags}**\n\n"
        f"_Replace this section with LLM synthesis and structured pros/cons._\n"
    )
    logger.info("Recommendation generated (stub)")
    return {
        "final_report": report,
        "trace": _append_trace(state, "recommendation", "stub markdown report"),
    }
