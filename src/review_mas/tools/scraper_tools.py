"""Tools for the Scraper/Data agent (local dataset ingestion).

These tools implement real-world interaction via local file I/O and strict validation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

from review_mas.state import ReviewRecord


@dataclass(frozen=True)
class CsvValidationError(ValueError):
    """Raised when the dataset CSV is missing required columns or has invalid rows."""

    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


REQUIRED_COLUMNS = ("review_id", "product_id", "rating", "review_text", "review_date")


def find_top_product_ids(dataset_path: str, top_k: int = 5) -> list[tuple[str, int]]:
    """Return the most frequent product_ids in a dataset CSV.

    This is used to pick a single product with many reviews for a realistic demo run.

    Args:
        dataset_path: Path to the CSV file.
        top_k: Number of product_ids to return.

    Returns:
        List of (product_id, count) sorted by count descending.
    """
    rows = load_and_validate_reviews_csv(dataset_path)
    counts = Counter((r.get("product_id") or "").strip() for r in rows if r.get("product_id"))
    return counts.most_common(top_k)


def filter_reviews_by_product_id(
    rows: list[ReviewRecord], product_id: str
) -> list[ReviewRecord]:
    """Filter already-loaded review records to a specific product_id."""
    pid = product_id.strip()
    return [r for r in rows if (r.get("product_id") or "").strip() == pid]


def load_and_validate_reviews_csv(dataset_path: str) -> list[ReviewRecord]:
    """Load reviews from a local CSV and validate schema.

    Expected columns:
      - review_id (string)
      - product_id (string)
      - rating (int; empty allowed -> 0)
      - review_text (string)
      - review_date (string; may be empty)

    Args:
        dataset_path: Path to the CSV file.

    Returns:
        List of normalized review records.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        CsvValidationError: If required columns are missing.
        ValueError: If a rating value cannot be parsed as an integer.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise CsvValidationError(
                f"CSV missing required columns: {missing}. Found: {list(header)}"
            )

        rows: list[ReviewRecord] = []
        for row in reader:
            rating_raw = row.get("rating")
            rating = int(rating_raw) if rating_raw not in (None, "") else 0
            rows.append(
                {
                    "review_id": (row.get("review_id") or "").strip(),
                    "product_id": (row.get("product_id") or "").strip(),
                    "rating": rating,
                    "review_text": (row.get("review_text") or "").strip(),
                    "review_date": (row.get("review_date") or "").strip(),
                }
            )

    return rows

