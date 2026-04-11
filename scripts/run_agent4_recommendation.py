"""Run the full pipeline through Agent 4 and write JSON (includes report_path).

Runs Scraper → Analysis → Fraud → Recommendation (same nodes as LangGraph).

Example:
  python scripts/run_agent4_recommendation.py --dataset data/sample_reviews.csv --out outputs/agent4.json

With Ollama for Analysis + Recommendation:
  python scripts/run_agent4_recommendation.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --out outputs/agent4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from review_mas.nodes import (  # noqa: E402
    analysis_node,
    fraud_node,
    recommendation_node,
    scraper_node,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--product-id", default="", dest="product_id")
    p.add_argument("--auto-product", action="store_true", dest="auto_product")
    p.add_argument("--use-ollama", action="store_true", dest="use_ollama")
    p.add_argument("--ollama-model", default="phi3", dest="ollama_model")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "agent4_recommendation.json",
    )
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "dataset_path": str(args.dataset),
        "product_id": args.product_id,
        "auto_product": bool(args.auto_product),
        "use_ollama": bool(args.use_ollama),
        "ollama_model": args.ollama_model,
        "trace": [],
    }
    for node in (scraper_node, analysis_node, fraud_node, recommendation_node):
        state.update(node(state))

    report = state.get("final_report") or ""
    payload = {
        "agents_run": ["scraper", "analysis", "fraud", "recommendation"],
        "outputs": {
            "report_path": state.get("report_path"),
            "final_report_preview": report[:2000] + ("…" if len(report) > 2000 else ""),
            "final_report_len": len(report),
            "analysis_summary": state.get("analysis_summary"),
            "fraud_summary": state.get("fraud_summary"),
            "trace_tail": (state.get("trace") or [])[-10:],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Markdown report: {state.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
