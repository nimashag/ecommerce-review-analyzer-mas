from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_scraper_tool_loads_valid_csv(tmp_path: Path):
    from review_mas.tools.scraper_tools import load_and_validate_reviews_csv

    p = tmp_path / "reviews.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "product_id", "rating", "review_text", "review_date"])
        w.writerow(["1", "p1", "5", "Nice product", "2024-01-01"])
        w.writerow(["2", "p1", "", "No rating provided", ""])

    rows = load_and_validate_reviews_csv(str(p))
    assert len(rows) == 2
    assert rows[0]["rating"] == 5
    assert rows[1]["rating"] == 0


def test_scraper_tool_rejects_missing_columns(tmp_path: Path):
    from review_mas.tools.scraper_tools import CsvValidationError, load_and_validate_reviews_csv

    p = tmp_path / "bad.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "rating", "review_text"])  # missing product_id, review_date
        w.writerow(["1", "5", "Ok"])

    with pytest.raises(CsvValidationError):
        load_and_validate_reviews_csv(str(p))


def test_find_top_product_ids(tmp_path: Path):
    from review_mas.tools.scraper_tools import find_top_product_ids

    p = tmp_path / "reviews.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "product_id", "rating", "review_text", "review_date"])
        w.writerow(["1", "p1", "5", "Nice product", "2024-01-01"])
        w.writerow(["2", "p1", "4", "Good", "2024-01-02"])
        w.writerow(["3", "p2", "1", "Bad", "2024-01-03"])

    top = find_top_product_ids(str(p), top_k=1)
    assert top == [("p1", 2)]


def test_scraper_tool_missing_file_raises():
    from review_mas.tools.scraper_tools import load_and_validate_reviews_csv

    with pytest.raises(FileNotFoundError):
        load_and_validate_reviews_csv("does_not_exist.csv")


def test_scraper_tool_invalid_rating_raises(tmp_path: Path):
    from review_mas.tools.scraper_tools import load_and_validate_reviews_csv

    p = tmp_path / "bad_rating.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "product_id", "rating", "review_text", "review_date"])
        w.writerow(["1", "p1", "not_an_int", "Nice product", "2024-01-01"])

    with pytest.raises(ValueError):
        load_and_validate_reviews_csv(str(p))


def test_filter_reviews_by_product_id():
    from review_mas.tools.scraper_tools import filter_reviews_by_product_id

    rows = [
        {"review_id": "1", "product_id": "p1", "rating": 5, "review_text": "x", "review_date": ""},
        {"review_id": "2", "product_id": "p2", "rating": 1, "review_text": "y", "review_date": ""},
        {"review_id": "3", "product_id": "p1", "rating": 4, "review_text": "z", "review_date": ""},
    ]
    filtered = filter_reviews_by_product_id(rows, "p1")
    assert [r["review_id"] for r in filtered] == ["1", "3"]


def test_scraper_node_auto_product_filters(tmp_path: Path):
    from review_mas.nodes import scraper_node

    p = tmp_path / "reviews.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "product_id", "rating", "review_text", "review_date"])
        w.writerow(["1", "p1", "5", "Nice product", "2024-01-01"])
        w.writerow(["2", "p1", "4", "Good", "2024-01-02"])
        w.writerow(["3", "p2", "1", "Bad", "2024-01-03"])

    out = scraper_node({"dataset_path": str(p), "auto_product": True})
    rows = out.get("raw_reviews") or []
    assert len(rows) == 2
    assert all(r.get("product_id") == "p1" for r in rows)

