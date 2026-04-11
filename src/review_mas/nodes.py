"""Agent steps (stubs). Replace with real tools + Ollama reasoning per teammate."""

from __future__ import annotations

import json
import logging

from review_mas.state import EnrichedReview, FraudAssessment, GraphState, ReviewRecord
from review_mas.tools.analysis_tools import (
    compute_review_statistics,
    extract_top_keywords,
    sentiment_label_from_rating,
    topics_for_review,
)
from review_mas.tools.fraud_tools import assess_fraud_patterns
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
    """Agent 2: deterministic stats + keywords; optional Ollama topic labels from facts only."""
    raw = list(state.get("raw_reviews") or [])
    stats = compute_review_statistics(raw)
    keyword_pairs = extract_top_keywords(raw, top_n=20)
    top_terms = [w for w, _ in keyword_pairs[:15]]

    enriched: list[EnrichedReview] = []
    for r in raw:
        rating = int(r.get("rating") or 0)
        label, score = sentiment_label_from_rating(rating)
        text = r.get("review_text") or ""
        topics = topics_for_review(text, top_terms, max_topics=5)
        enriched.append(
            {
                **r,
                "sentiment_label": label,
                "sentiment_score": score,
                "topics": topics,
            }
        )

    analysis_summary: dict = {
        **stats,
        "top_keywords": [{"term": w, "count": c} for w, c in keyword_pairs[:10]],
    }

    trace_detail = (
        f"tool:compute_review_statistics+extract_top_keywords "
        f"n={stats.get('n_reviews')} avg_rating={stats.get('avg_rating')}"
    )

    if state.get("use_ollama", False):
        model = state.get("ollama_model") or "phi3"
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=model, temperature=0.1)
            facts_json = json.dumps(analysis_summary, ensure_ascii=False)
            prompt = (
                "You label product-review themes for a buyer report.\n"
                "Rules:\n"
                "- Use ONLY the JSON facts below (counts, histogram, keyword list).\n"
                "- Do NOT invent percentages, prices, shipping claims, or product specs.\n"
                "- Output EXACTLY 5 lines, each line one short topic label (2-5 words), no numbering.\n\n"
                f"FACTS_JSON:\n{facts_json}\n"
            )
            msg = llm.invoke(
                [
                    SystemMessage(
                        content="Follow output format strictly. No markdown fences."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw_text = str(getattr(msg, "content", "")).strip()
            lines = [ln.strip("- ").strip() for ln in raw_text.splitlines() if ln.strip()]
            labels = [ln for ln in lines if ln][:5]
            if labels:
                analysis_summary["llm_topic_labels"] = labels
                trace_detail += f" ollama:{model} topic_labels={len(labels)}"
        except Exception as e:  # noqa: BLE001
            logger.warning("Analysis Ollama topic step failed (continuing without): %s", e)
            trace_detail += " ollama_topic_failed=true"

    logger.info("Analysis enriched %d reviews", len(enriched))
    return {
        "enriched_reviews": enriched,
        "analysis_summary": analysis_summary,
        "trace": _append_trace(state, "analysis", trace_detail),
    }


def fraud_node(state: GraphState) -> dict:
    """Agent 3: fraud / spam heuristics via tools; optional Ollama overview from facts only."""
    enriched = list(state.get("enriched_reviews") or [])
    assessments, fraud_summary = assess_fraud_patterns(enriched)
    flagged_n = int(fraud_summary.get("flagged_count") or 0)

    trace_detail = (
        f"tool:assess_fraud_patterns flagged={flagged_n} "
        f"avg_score={fraud_summary.get('avg_fraud_score')}"
    )

    if state.get("use_ollama", False):
        model = state.get("ollama_model") or "phi3"
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=model, temperature=0.1)
            flagged_rows = [a for a in assessments if a.get("flagged")][:8]
            payload = {
                "fraud_summary": fraud_summary,
                "flagged_sample": flagged_rows,
            }
            facts_json = json.dumps(payload, ensure_ascii=False)
            prompt = (
                "You explain fraud/spam risk for shoppers based ONLY on the JSON below.\n"
                "Rules:\n"
                "- Do NOT invent counts; repeat only numbers present in fraud_summary.\n"
                "- Write EXACTLY 4 bullet lines starting with '- '.\n"
                "- Each bullet max 20 words.\n\n"
                f"FACTS_JSON:\n{facts_json}\n"
            )
            msg = llm.invoke(
                [
                    SystemMessage(content="No markdown fences. No extra sections."),
                    HumanMessage(content=prompt),
                ]
            )
            txt = str(getattr(msg, "content", "")).strip()
            if txt:
                fraud_summary["llm_fraud_overview"] = txt
                trace_detail += f" ollama:{model} fraud_overview_len={len(txt)}"
        except Exception as e:  # noqa: BLE001
            logger.warning("Fraud Ollama overview failed (continuing without): %s", e)
            trace_detail += " ollama_fraud_failed=true"

    logger.info("Fraud assessed %d reviews, %d flagged", len(assessments), flagged_n)
    return {
        "fraud_assessments": assessments,
        "fraud_summary": fraud_summary,
        "trace": _append_trace(state, "fraud", trace_detail),
    }


def recommendation_node(state: GraphState) -> dict:
    """Agent 4: buyer report (uses Ollama when enabled); writes Markdown via tool."""
    from review_mas.tools.recommendation_tools import write_markdown_report

    n = len(state.get("enriched_reviews") or [])
    flags = sum(1 for a in state.get("fraud_assessments") or [] if a.get("flagged"))
    summary = state.get("analysis_summary") or {}
    fraud_s = state.get("fraud_summary") or {}
    facts_json = json.dumps(summary, ensure_ascii=False)
    fraud_json = json.dumps(fraud_s, ensure_ascii=False)

    trace_detail: str

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
                "Rules:\n"
                "- Use ONLY the numbers and facts in FACTS_JSON, FRAUD_JSON, and the two bullet counts below.\n"
                "- Do NOT invent percentages, prices, shipping claims, specs, or review counts not shown.\n"
                "- If a fact is missing, say \"not available\" instead of guessing.\n"
                "- Use fraud_summary.flagged_count and reason_counts for red flags (do not guess).\n\n"
                f"- Reviews in this run: {n}\n"
                f"- Fraud assessments flagged: {flags}\n\n"
                f"FACTS_JSON:\n{facts_json}\n\n"
                f"FRAUD_JSON:\n{fraud_json}\n"
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
            trace_detail = f"ollama:{model} report_len={len(report)}"
        except Exception as e:  # noqa: BLE001
            logger.exception("Ollama recommendation failed")
            raise RuntimeError(
                "Ollama call failed. Ensure Ollama is installed and running, and the model is pulled.\n"
                "Example:\n"
                "  ollama pull phi3\n"
                "  ollama serve\n"
                f"Original error: {e}"
            ) from e
    else:
        avg = summary.get("avg_rating", "n/a")
        fc = fraud_s.get("flagged_count", "n/a")
        report = (
            f"## Should you buy? (stub)\n\n"
            f"- Reviews analyzed: **{n}**\n"
            f"- Avg rating (from data): **{avg}**\n"
            f"- Fraud flagged (from fraud_summary): **{fc}** (per-review flags: **{flags}**)\n\n"
            f"_Run with --use-ollama to generate an Ollama-backed report._\n"
        )
        logger.info("Recommendation generated (stub; Ollama disabled)")
        trace_detail = "stub markdown report (ollama disabled)"

    report_path = write_markdown_report("final_buyer_report.md", report)
    logger.info("Buyer report written to %s", report_path)
    trace_detail = f"{trace_detail}; tool:write_markdown_report path={report_path}"

    return {
        "final_report": report,
        "report_path": report_path,
        "trace": _append_trace(state, "recommendation", trace_detail),
    }
