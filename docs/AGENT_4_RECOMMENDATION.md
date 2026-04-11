## Agent 4 — Buyer recommendation

Agent 4 turns **structured outputs** from Agents 2–3 into a **Markdown buyer report**. It uses **only grounded facts** (`analysis_summary`, `fraud_summary`, counts) in the LLM prompt when Ollama is enabled; otherwise it emits a short **stub** report. Every run **persists** the report to disk via a custom tool.

---

## Tools

File: `src/review_mas/tools/recommendation_tools.py`

| Function | Role |
|----------|------|
| `write_markdown_report(relative_path, content, reports_dir="reports")` | Writes UTF-8 Markdown under `reports/` (cwd-relative). Rejects absolute paths and `..` traversal outside `reports_dir`. |

The graph node calls `write_markdown_report("final_buyer_report.md", report)` after the body is built (Ollama or stub).

---

## State outputs

- `final_report`: full Markdown string (also printed by `run.py` under `--- final_report ---`)
- `report_path`: absolute path to the file written by the tool
- `trace`: includes `tool:write_markdown_report path=...` in the recommendation step detail

---

## Run full pipeline (includes Agent 4)

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3
```

The Markdown file is created at `reports/final_buyer_report.md` (from repo root). With `--show-agent-outputs`, Agent 4 prints `report_path` and a preview of `final_report`.

---

## Run through Agent 4 only (JSON metadata)

```powershell
python scripts/run_agent4_recommendation.py --dataset data/sample_reviews.csv --out outputs/agent4.json
```

With Ollama:

```powershell
python scripts/run_agent4_recommendation.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --out outputs/agent4.json
```

---

## Tests

`tests/test_recommendation_tools.py` — write success, path traversal rejection, absolute path rejection.

`tests/test_pipeline.py` — smoke run asserts `report_path` exists on disk and matches `final_report`.
