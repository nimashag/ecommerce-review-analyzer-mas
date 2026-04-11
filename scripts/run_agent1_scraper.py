"""Run Agent 1 (Scraper) only and write outputs to a JSON file.

Example:
  python scripts/run_agent1_scraper.py --dataset data/hf_reviews_electronics.csv --auto-product --out outputs/agent1_scraper.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from review_mas.nodes import scraper_node  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Path to CSV")
    p.add_argument("--product-id", default="", dest="product_id")
    p.add_argument("--auto-product", action="store_true", dest="auto_product")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "agent1_scraper.json",
        help="Output JSON path",
    )
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    state_in = {
        "dataset_path": str(args.dataset),
        "product_id": args.product_id,
        "auto_product": bool(args.auto_product),
        "trace": [],
    }
    patch = scraper_node(state_in)
    merged = {**state_in, **patch}
    payload = {
        "agent": "scraper",
        "outputs": {
            "raw_reviews": merged.get("raw_reviews"),
            "trace_tail": (merged.get("trace") or [])[-3:],
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
