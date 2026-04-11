"""Smoke test: pipeline runs and produces a report."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

from review_mas.graph import run_pipeline  # noqa: E402


def test_pipeline_smoke():
    state = run_pipeline(
        {
            "dataset_path": str(_ROOT / "data" / "sample_reviews.csv"),
            "use_ollama": False,
        }
    )
    assert state.get("raw_reviews")
    assert state.get("enriched_reviews")
    summary = state.get("analysis_summary") or {}
    assert summary.get("n_reviews") == len(state.get("raw_reviews") or [])
    assert "avg_rating" in summary
    assert isinstance(state.get("fraud_summary"), dict)
    assert state.get("fraud_assessments") is not None
    assert state.get("final_report")
    rp = state.get("report_path")
    assert rp
    assert Path(rp).is_file()
    assert Path(rp).read_text(encoding="utf-8") == state["final_report"]
    assert isinstance(state.get("trace"), list)
