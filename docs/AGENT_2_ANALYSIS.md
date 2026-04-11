## Agent 2 — Analysis (insights per review + grounded summary)

This document explains **Agent 2 (Analysis)** for the CTSE Assignment 2 MAS. It is written to align with the rubric: **tools first**, **state handoff**, **observability**, and **local Ollama only as a formatter** (not as the source of numerical facts).

---

## What this agent does

1. **Computes deterministic facts** from `raw_reviews` (no cloud, no paid APIs):
   - rating histogram (1–5)
   - average rating (from rated reviews only)
   - rule-based sentiment buckets (positive / neutral / negative) derived from star ratings
   - top keywords across all review text (simple token frequency)

2. **Enriches each review** (`enriched_reviews`):
   - `sentiment_label`, `sentiment_score` (coarse, rule-based)
   - `topics`: up to 5 keywords from the global “top terms” list that actually appear in that review’s text

3. **Writes a grounded global summary** into state:
   - `analysis_summary`: JSON-serializable dict used by **Recommendation** (and future Fraud explanations)

4. **Optional Ollama (only if `--use-ollama`)**:
   - Sends **only** `analysis_summary` as JSON to the local model
   - Asks for **5 short topic labels** (no invented numbers)
   - Stores result under `analysis_summary["llm_topic_labels"]` if successful
   - If Ollama fails, Analysis continues without LLM labels (tool outputs still valid)

---

## Tools (custom Python)

Implemented in:

- `src/review_mas/tools/analysis_tools.py`

Key functions:

| Function | Purpose |
|----------|---------|
| `compute_review_statistics(reviews)` | Histogram, averages, sentiment bucket counts |
| `extract_top_keywords(reviews, top_n)` | Top terms from review text |
| `sentiment_label_from_rating(rating)` | Maps stars → label + coarse score |
| `topics_for_review(text, top_terms, max_topics)` | Per-review keyword overlap |

---

## State management (outputs)

Agent 2 writes:

- `enriched_reviews`: list used by Fraud + Recommendation
- `analysis_summary`: dict passed forward (grounded facts)

Downstream agents should treat `analysis_summary` as the **authoritative numeric summary**.

---

## Observability

The Analysis trace entry includes:

- tool usage summary: `tool:compute_review_statistics+extract_top_keywords n=... avg_rating=...`
- optional: `ollama:<model> topic_labels=5` or `ollama_topic_failed=true`

---

## How to run

### Run **only** Agent 2 (Analysis) and save output to a JSON file

Analysis needs `raw_reviews` from the scraper first; this script runs scraper then analysis:

```powershell
python scripts/run_agent2_analysis.py --dataset data/hf_reviews_electronics.csv --auto-product --out outputs/agent2_analysis.json
```

With Ollama for Analysis topic labels:

```powershell
python scripts/run_agent2_analysis.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --out outputs/agent2_analysis.json
```

Open `outputs/agent2_analysis.json` for `analysis_summary` and a small sample of `enriched_reviews`.

---

Same entry point as the full MAS:

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product
```

With Ollama (Analysis + Recommendation use local model when this flag is on):

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3
```

Print **sample outputs from every agent** after the run (raw rows, `analysis_summary`, enriched, fraud):

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --show-agent-outputs --sample-reviews 5
```

Save terminal output (requires `outputs/` — included in repo):

```powershell
python run.py ... --show-agent-outputs | Tee-Object -FilePath outputs\last_run.txt
```

---

## Tests

```powershell
python -m pytest tests/test_analysis_agent.py -v
python -m pytest -q
```

Tests validate:

- histogram + sentiment bucket counts
- keyword ordering
- per-review topic overlap behavior

---

## Assignment alignment (why this design)

- **Tool usage**: Analysis relies on custom Python tools for facts (not “LLM-only”).
- **Ollama constraint**: Ollama is optional and constrained to **reformat/label** using provided JSON facts.
- **State**: `analysis_summary` is explicit global state for the next agent.
- **Security / accuracy**: Recommendation prompt is updated to **forbid inventing** numbers not present in `FACTS_JSON`.
