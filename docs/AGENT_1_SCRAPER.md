## Agent 1 — Scraper / Data Ingestion

This document explains **Agent 1 (Scraper)** in the Smart Product Review Analysis MAS:

- What it does
- Which tools it uses (custom Python tools)
- What it outputs into global state
- How to run it (including product filtering)
- How to demo it for the assignment rubric

---

## What this agent does (high level)

Agent 1 is the **Data Ingestion + Validation** agent. Its job is to make sure downstream agents (Analysis/Fraud/Recommendation) receive **clean, trusted input**.

It performs:

- **Read local dataset** (CSV)
- **Validate schema** (required columns exist)
- **Normalize rows** (trim whitespace, parse ratings, handle blanks safely)
- **(Optional) focus on one product** (so “Should you buy?” is meaningful)

---

## Inputs

- **dataset_path**: path to a CSV file with these required columns:
  - `review_id`
  - `product_id`
  - `rating`
  - `review_text`
  - `review_date`

Optional:

- **product_id**: filter all loaded reviews to one product
- **auto_product**: automatically pick the product_id with the most reviews in the dataset

---

## Outputs (where the data goes)

The output is placed into the **LangGraph global state** as:

- **raw_reviews**: a list of review records (Python dicts) that downstream agents use
- **trace**: a trace entry describing the tool call and the selected product_id

This output is **not automatically written to a file**—it is passed in-memory through the graph state.

---

## Tools used by this agent (custom Python tools)

These tools are implemented in:

- `src/review_mas/tools/scraper_tools.py`

Main tool:

- **load_and_validate_reviews_csv(dataset_path)**
  - Reads the CSV from disk (real-world interaction)
  - Validates required columns
  - Normalizes rows (rating parsing, trimming)

Helper tools (support realistic product-specific runs):

- **find_top_product_ids(dataset_path, top_k)**
  - Counts reviews per product_id
  - Returns the top-k products with the most reviews

- **filter_reviews_by_product_id(rows, product_id)**
  - Filters already-loaded rows to one product_id

---

## How to run (commands)

From repo root:

### 1) Export a realistic Electronics dataset (recommended)

This downloads from Hugging Face once (streaming) and writes a local CSV snapshot:

```powershell
python scripts/export_hf_reviews.py --limit 50000 --output data/hf_reviews_electronics.csv
```

### 2) List top products in the dataset (pick a product with many reviews)

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --list-top-products 10
```

### 3) Run the pipeline on a single product (best demo mode)

Auto-pick the product with the most reviews:

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product
```

Or choose a specific product_id:

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --product-id B075X8471B
```

### 4) (Optional) Run with Ollama too (full assignment demo)

```powershell
python run.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3
```

---

## How this agent supports the assignment rubric

- **Tool Usage**: Reads local files + validates schema using custom tools (not “LLM-only”).
- **State Management**: Outputs `raw_reviews` into LangGraph global state for the next agent.
- **Observability**: Adds a `trace` entry like:
  - `tool:load_and_validate_reviews_csv rows=166 product_id=B075X8471B auto_product=true`

---

## Testing / evaluation

Automated tests for the Scraper tool are in:

- `tests/test_scraper_agent.py`

### How to run the tests (commands)

From repo root:

Run only Scraper tests (quiet):

```powershell
python -m pytest tests/test_scraper_agent.py -q
```

Run only Scraper tests (detailed):

```powershell
python -m pytest tests/test_scraper_agent.py -v
```

Run all tests in the project:

```powershell
python -m pytest -q
```

They verify:

- Valid CSV loads successfully
- Missing required columns triggers validation error
- Top product detection returns expected counts

