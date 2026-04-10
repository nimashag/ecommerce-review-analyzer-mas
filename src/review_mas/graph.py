"""LangGraph orchestration: Scraper → Analysis → Fraud → Recommendation."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from review_mas.nodes import analysis_node, fraud_node, recommendation_node, scraper_node
from review_mas.state import GraphState

logger = logging.getLogger(__name__)


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("scraper", scraper_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("fraud", fraud_node)
    workflow.add_node("recommendation", recommendation_node)

    workflow.set_entry_point("scraper")
    workflow.add_edge("scraper", "analysis")
    workflow.add_edge("analysis", "fraud")
    workflow.add_edge("fraud", "recommendation")
    workflow.add_edge("recommendation", END)

    return workflow.compile()


def run_pipeline(initial: GraphState | None = None) -> GraphState:
    """Execute full multi-agent workflow and return final state."""
    initial = initial or {"dataset_path": "data/sample_reviews.csv"}
    graph = build_graph()
    logger.info("Starting MAS pipeline")
    result = graph.invoke(initial)
    return result
