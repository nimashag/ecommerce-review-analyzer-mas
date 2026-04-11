from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


def test_write_markdown_report_creates_file(tmp_path: Path) -> None:
    from review_mas.tools.recommendation_tools import write_markdown_report

    reports = tmp_path / "reports"
    body = "# Title\n\nHello **world**.\n"
    out = write_markdown_report("buyer.md", body, reports_dir=str(reports))
    p = Path(out)
    assert p.is_file()
    assert p.read_text(encoding="utf-8") == body


def test_write_markdown_report_rejects_path_traversal(tmp_path: Path) -> None:
    from review_mas.tools.recommendation_tools import ReportPathError, write_markdown_report

    reports = tmp_path / "reports"
    with pytest.raises(ReportPathError):
        write_markdown_report("../escape.md", "x", reports_dir=str(reports))


def test_write_markdown_report_rejects_absolute_path(tmp_path: Path) -> None:
    from review_mas.tools.recommendation_tools import ReportPathError, write_markdown_report

    reports = tmp_path / "reports"
    with pytest.raises(ReportPathError):
        write_markdown_report(str(tmp_path / "evil.md"), "x", reports_dir=str(reports))
