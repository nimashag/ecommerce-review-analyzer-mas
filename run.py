"""Entry point: configure logging and run the four-agent pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python run.py` from repo root without installing the package
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from review_mas.graph import run_pipeline  # noqa: E402


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
        help="Generate the final recommendation using a local Ollama model.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="phi3",
        help="Ollama model name (example: phi3, llama3.1:8b).",
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


if __name__ == "__main__":
    main()
