"""Entry point: configure logging and run the four-agent pipeline."""

from __future__ import annotations

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    state = run_pipeline({"dataset_path": "data/sample_reviews.csv"})
    print("\n--- final_report ---\n")
    print(state.get("final_report", ""))
    print("\n--- trace (observability stub) ---")
    for step in state.get("trace") or []:
        print(f"  {step}")


if __name__ == "__main__":
    main()
