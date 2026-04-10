# E-Commerce Review Analyzer MAS

Multi-agent system (CTSE Assignment 2): scrape/load reviews → analyze → fraud signals → buyer recommendation. Runs **locally** with **Ollama** (no paid LLM APIs).

## Prerequisites

- Python **3.9+** (this repo pins LangGraph 0.2.x for broad compatibility; **3.10+** recommended if you upgrade to LangGraph 0.6+ later)
- [Ollama](https://ollama.com) installed; pull a model, e.g. `ollama pull llama3.1:8b`

## Setup

```powershell
cd ecommerce-review-analyzer-mas
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (stub pipeline)

```powershell
python run.py
```

This executes the four-agent **LangGraph** workflow with placeholder logic. Replace stubs with real tools and Ollama calls per agent.

## Layout

| Path | Purpose |
|------|--------|
| `src/review_mas/state.py` | Shared state passed between agents |
| `src/review_mas/graph.py` | LangGraph workflow |
| `src/review_mas/nodes.py` | Agent step functions (stubs → real) |
| `src/review_mas/tools/` | Custom Python tools (one+ per teammate) |
| `data/sample_reviews.csv` | Sample local dataset |
| `tests/` | Pytest harness (expand per agent) |

## Team

Four agents: **Scraper** → **Analysis** → **Fraud** → **Recommendation** — one primary owner each for agent design, tool, and tests.
