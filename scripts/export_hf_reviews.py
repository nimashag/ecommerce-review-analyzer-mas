"""Export a Hugging Face dataset slice to local CSV for the MAS scraper.

Run once (needs internet); after that the MAS reads only the local CSV file.

Default preset is a **realistic Electronics** review dataset:
  - McAuley-Lab/Amazon-Reviews-2023, config raw_review_Electronics

Example:
  python scripts/export_hf_reviews.py --limit 3000 --output data/hf_reviews_electronics.csv

Notes:
- Some Amazon review datasets are extremely large. This script uses streaming to export only
  the first N rows without downloading the entire dataset.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = ["review_id", "product_id", "rating", "review_text", "review_date"]


def _ms_to_iso_date(ms: int | None) -> str:
    """Convert epoch milliseconds to YYYY-MM-DD (UTC)."""
    if ms is None:
        return ""
    try:
        d = dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc).date()
        return d.isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _write_rows(output_path: Path, rows: Iterable[dict[str, str | int]]) -> int:
    """Write mapped rows to CSV. Returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for mapped in rows:
            writer.writerow(mapped)
            count += 1
    return count


def export_amazon_reviews_2023_electronics(
    output_path: Path,
    limit: int = 3000,
    dataset_id: str = "McAuley-Lab/Amazon-Reviews-2023",
    config: str = "raw_review_Electronics",
    split: str = "full",
) -> int:
    """Stream Amazon-Reviews-2023 Electronics reviews and write a local CSV snapshot."""
    from datasets import load_dataset  # noqa: PLC0415 — heavy import

    ds = load_dataset(
        dataset_id,
        config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    def mapped_rows() -> Iterator[dict[str, str | int]]:
        for i, row in enumerate(ds):
            if i >= limit:
                break
            title = (row.get("title") or "").strip()
            body = (row.get("text") or "").strip()
            text = f"{title}. {body}".strip() if title else body
            rating_raw = row.get("rating", 0)
            rating = int(round(float(rating_raw))) if rating_raw not in ("", None) else 0
            product_id = (row.get("parent_asin") or row.get("asin") or "unknown").strip()
            yield {
                "review_id": str(i),
                "product_id": product_id,
                "rating": max(0, min(5, rating)),
                "review_text": text,
                "review_date": _ms_to_iso_date(row.get("timestamp")),
            }

    return _write_rows(output_path, mapped_rows())


def export_amazon_polarity(output_path: Path, limit: int = 3000) -> int:
    """Legacy fallback: export amazon_polarity slice (no real product metadata)."""
    from datasets import load_dataset  # noqa: PLC0415 — heavy import

    ds = load_dataset("amazon_polarity", split=f"train[:{limit}]")

    def mapped_rows() -> Iterator[dict[str, str | int]]:
        for i, row in enumerate(ds):
            title = (row.get("title") or "").strip()
            body = (row.get("content") or "").strip()
            text = f"{title}. {body}".strip() if title else body
            label = int(row.get("label", 0))
            yield {
                "review_id": str(i),
                "product_id": "amazon_polarity",
                "rating": 5 if label == 1 else 1,
                "review_text": text,
                "review_date": "",
            }

    return _write_rows(output_path, mapped_rows())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "hf_reviews_electronics.csv",
        help="Output CSV path (default: data/hf_reviews_electronics.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Max rows to export (default: 3000).",
    )
    parser.add_argument(
        "--dataset",
        default="amazon_reviews_2023_electronics",
        choices=["amazon_reviews_2023_electronics", "amazon_polarity"],
        help="Which dataset preset to export.",
    )
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output

    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")

    if args.dataset == "amazon_reviews_2023_electronics":
        n = export_amazon_reviews_2023_electronics(out, limit=args.limit)
    else:
        n = export_amazon_polarity(out, limit=args.limit)

    print(f"Wrote {n} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
