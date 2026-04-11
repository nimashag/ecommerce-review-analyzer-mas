"""Run Agents 1–3 only (Scraper → Analysis → Fraud) and write JSON output.

Example:
  python scripts/run_agent3_fraud.py --dataset data/hf_reviews_electronics.csv --auto-product --out outputs/agent3_fraud.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from review_mas.nodes import analysis_node, fraud_node, scraper_node  # noqa: E402


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
        default=ROOT / "outputs" / "agent3_fraud.json",
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
    state.update(scraper_node(state))
    state.update(analysis_node(state))
    state.update(fraud_node(state))

    payload = {
        "agents_run": ["scraper", "analysis", "fraud"],
        "outputs": {
            "fraud_summary": state.get("fraud_summary"),
            "fraud_assessments_sample": (state.get("fraud_assessments") or [])[:8],
            "fraud_assessments_count": len(state.get("fraud_assessments") or []),
            "trace_tail": (state.get("trace") or [])[-8:],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
