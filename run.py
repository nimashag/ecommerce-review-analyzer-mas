"""Entry point: configure logging and run the four-agent pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python run.py` from repo root without installing the package
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from review_mas.graph import run_pipeline  # noqa: E402


def _print_agent_outputs(state: dict, sample: int) -> None:
    """Print a readable slice of each agent's outputs (final state)."""
    sample = max(0, min(sample, 50))

    print("\n=== Agent 1 (Scraper) — raw_reviews ===\n")
    raw = state.get("raw_reviews") or []
    print(f"count: {len(raw)}")
    for row in raw[:sample]:
        print(json.dumps(row, ensure_ascii=False))

    print("\n=== Agent 2 (Analysis) — analysis_summary (grounded JSON) ===\n")
    summary = state.get("analysis_summary") or {}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Agent 2 (Analysis) — enriched_reviews (sample) ===\n")
    enriched = state.get("enriched_reviews") or []
    print(f"count: {len(enriched)}")
    for row in enriched[:sample]:
        print(json.dumps(row, ensure_ascii=False))

    print("\n=== Agent 3 (Fraud) — fraud_assessments (sample) ===\n")
    fraud = state.get("fraud_assessments") or []
    print(f"count: {len(fraud)}")
    flagged = sum(1 for a in fraud if a.get("flagged"))
    print(f"flagged_count: {flagged}")
    for row in fraud[:sample]:
        print(json.dumps(row, ensure_ascii=False))

    print("\n=== Agent 3 (Fraud) — fraud_summary (aggregate JSON) ===\n")
    print(json.dumps(state.get("fraud_summary") or {}, indent=2, ensure_ascii=False))

    print("\n=== Agent 4 (Recommendation) — report_path ===\n")
    print(state.get("report_path") or "(not set)")

    print("\n=== Agent 4 (Recommendation) — final_report (preview) ===\n")
    report = state.get("final_report") or ""
    preview = report[:1200] + ("…\n" if len(report) > 1200 else "")
    print(preview or "(empty)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run review MAS pipeline locally.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_reviews.csv",
        help="Path to CSV (review_id, product_id, rating, review_text, review_date)",
    )
    parser.add_argument(
        "--product-id",
        type=str,
        default="",
        help="Filter to a single product_id (recommended for realistic 'Should you buy?' reports).",
    )
    parser.add_argument(
        "--auto-product",
        action="store_true",
        help="Auto-pick the most reviewed product_id in the dataset and run the pipeline only for it.",
    )
    parser.add_argument(
        "--list-top-products",
        type=int,
        default=0,
        help="Print the top N product_ids by review count and exit (0 disables).",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama for Analysis (topic labels) and Recommendation (final report).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="phi3",
        help="Ollama model name (example: phi3, llama3.1:8b).",
    )
    parser.add_argument(
        "--show-agent-outputs",
        action="store_true",
        help="After the run, print sample outputs from each agent (raw, analysis JSON, enriched, fraud, report).",
    )
    parser.add_argument(
        "--sample-reviews",
        type=int,
        default=3,
        help="With --show-agent-outputs, how many review rows to print per sample block (default: 3).",
    )
    args = parser.parse_args()

    if args.list_top_products and args.list_top_products > 0:
        from review_mas.tools.scraper_tools import find_top_product_ids

        top = find_top_product_ids(args.dataset, top_k=args.list_top_products)
        print("\n--- top products ---\n")
        for pid, count in top:
            print(f"{pid}\t{count}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    state = run_pipeline(
        {
            "dataset_path": args.dataset,
            "product_id": args.product_id,
            "auto_product": bool(args.auto_product),
            "use_ollama": bool(args.use_ollama),
            "ollama_model": args.ollama_model,
        }
    )
    print("\n--- final_report ---\n")
    print(state.get("final_report", ""))
    print("\n--- trace (observability stub) ---")
    for step in state.get("trace") or []:
        print(f"  {step}")

    if args.show_agent_outputs:
        _print_agent_outputs(state, sample=args.sample_reviews)


if __name__ == "__main__":
    main()
