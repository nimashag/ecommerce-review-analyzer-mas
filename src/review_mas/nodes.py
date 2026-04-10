"""Agent steps (stubs). Replace with real tools + Ollama reasoning per teammate."""

from __future__ import annotations

import logging

from review_mas.state import EnrichedReview, FraudAssessment, GraphState, ReviewRecord
from review_mas.tools.scraper_tools import (
    CsvValidationError,
    filter_reviews_by_product_id,
    find_top_product_ids,
    load_and_validate_reviews_csv,
)

logger = logging.getLogger(__name__)


def _append_trace(state: GraphState, agent: str, detail: str) -> list[dict]:
    trace = list(state.get("trace") or [])
    trace.append({"agent": agent, "detail": detail})
    return trace


def scraper_node(state: GraphState) -> dict:
    """Agent 1: load and validate raw reviews from a local CSV (tool-based)."""
    dataset_path = state.get("dataset_path") or "data/sample_reviews.csv"
    try:
        rows: list[ReviewRecord] = load_and_validate_reviews_csv(dataset_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return {
            "raw_reviews": [],
            "trace": _append_trace(state, "scraper", f"missing file: {dataset_path}"),
        }
    except CsvValidationError as e:
        logger.error(str(e))
        return {
            "raw_reviews": [],
            "trace": _append_trace(state, "scraper", f"invalid csv: {e}"),
        }

    chosen_product_id = (state.get("product_id") or "").strip()
    if state.get("auto_product", False) and not chosen_product_id:
        top = find_top_product_ids(dataset_path, top_k=1)
        if top:
            chosen_product_id = top[0][0]

    if chosen_product_id:
        rows = filter_reviews_by_product_id(rows, chosen_product_id)

    logger.info(
        "Scraper loaded %d reviews from %s%s",
        len(rows),
        dataset_path,
        f" (product_id={chosen_product_id})" if chosen_product_id else "",
    )
    return {
        "raw_reviews": rows,
        "trace": _append_trace(
            state,
            "scraper",
            (
                f"tool:load_and_validate_reviews_csv rows={len(rows)}"
                + (f" product_id={chosen_product_id}" if chosen_product_id else "")
                + (" auto_product=true" if state.get("auto_product", False) else "")
            ),
        ),
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
    """Agent 4: buyer report (uses Ollama when enabled)."""
    n = len(state.get("enriched_reviews") or [])
    flags = sum(1 for a in state.get("fraud_assessments") or [] if a.get("flagged"))

    if state.get("use_ollama", False):
        model = state.get("ollama_model") or "phi3"
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOllama(model=model, temperature=0.2)
            prompt = (
                "You are a buyer assistant for an e-commerce review analysis system.\n"
                "Write a concise recommendation report in Markdown with these sections:\n"
                "## Should you buy?\n"
                "## Pros\n"
                "## Cons\n"
                "## Red flags\n"
                "## Confidence\n\n"
                "Use only the provided summary. Do not invent product features.\n\n"
                f"Summary:\n- Reviews analyzed: {n}\n- Suspicious flags: {flags}\n"
            )
            msg = llm.invoke(
                [
                    SystemMessage(
                        content="Follow instructions strictly. Be brief and factual."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            report = str(getattr(msg, "content", "")).strip()
            if not report:
                raise RuntimeError("Ollama returned empty content")

            logger.info("Recommendation generated with Ollama model=%s", model)
            return {
                "final_report": report,
                "trace": _append_trace(
                    state, "recommendation", f"ollama:{model} report_len={len(report)}"
                ),
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("Ollama recommendation failed")
            raise RuntimeError(
                "Ollama call failed. Ensure Ollama is installed and running, and the model is pulled.\n"
                "Example:\n"
                "  ollama pull phi3\n"
                "  ollama serve\n"
                f"Original error: {e}"
            ) from e

    report = (
        f"## Should you buy? (stub)\n\n"
        f"- Reviews analyzed: **{n}**\n"
        f"- Suspicious / duplicate-text flags: **{flags}**\n\n"
        f"_Run with --use-ollama to generate an Ollama-backed report._\n"
    )
    logger.info("Recommendation generated (stub; Ollama disabled)")
    return {
        "final_report": report,
        "trace": _append_trace(state, "recommendation", "stub markdown report (ollama disabled)"),
    }
