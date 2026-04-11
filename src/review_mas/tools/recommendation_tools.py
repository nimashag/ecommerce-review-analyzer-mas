"""Tools for the Recommendation agent (write buyer-facing artifacts to disk)."""

from __future__ import annotations

from pathlib import Path


class ReportPathError(ValueError):
    """Raised when the requested report path escapes the allowed directory."""


def write_markdown_report(relative_path: str, content: str, reports_dir: str = "reports") -> str:
    """Write Markdown report text under a dedicated ``reports/`` directory.

    The path is restricted to stay inside ``reports_dir`` (resolved relative to the
    process current working directory — run the CLI from the repo root).

    Args:
        relative_path: File name or subpath under ``reports_dir`` (e.g. ``final_buyer_report.md``).
        content: Full Markdown body to write (UTF-8).
        reports_dir: Root folder for buyer reports (created if missing).

    Returns:
        Absolute path to the written file as a string.

    Raises:
        ReportPathError: If ``relative_path`` tries to escape ``reports_dir`` (e.g. ``..`` segments).
        OSError: If the file cannot be written.
    """
    root = Path(reports_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    rel = Path(relative_path)
    if rel.is_absolute():
        raise ReportPathError("relative_path must not be absolute")

    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ReportPathError(
            f"Report path escapes allowed directory: {target} not under {root}"
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)
