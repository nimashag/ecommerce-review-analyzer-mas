"""Run Agent 1 then Agent 2 only; write Analysis outputs to a JSON file.

Agent 2 needs ``raw_reviews`` from the scraper. This script runs scraper_node then analysis_node.

Example:
  python scripts/run_agent2_analysis.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --out outputs/agent2_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from review_mas.nodes import analysis_node, scraper_node  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Path to CSV")
    p.add_argument("--product-id", default="", dest="product_id")
    p.add_argument("--auto-product", action="store_true", dest="auto_product")
    p.add_argument("--use-ollama", action="store_true", dest="use_ollama")
    p.add_argument("--ollama-model", default="phi3", dest="ollama_model")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "agent2_analysis.json",
        help="Output JSON path",
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
    s1 = scraper_node(state)
    state = {**state, **s1}
    s2 = analysis_node(state)
    state = {**state, **s2}

    payload = {
        "agents_run": ["scraper", "analysis"],
        "outputs": {
            "analysis_summary": state.get("analysis_summary"),
            "enriched_reviews_sample": (state.get("enriched_reviews") or [])[:5],
            "enriched_count": len(state.get("enriched_reviews") or []),
            "trace_tail": (state.get("trace") or [])[-6:],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
