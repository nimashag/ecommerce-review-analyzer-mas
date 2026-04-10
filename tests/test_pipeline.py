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
    assert state.get("fraud_assessments") is not None
    assert state.get("final_report")
    assert isinstance(state.get("trace"), list)
