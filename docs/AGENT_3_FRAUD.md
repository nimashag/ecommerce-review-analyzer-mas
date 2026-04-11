## Agent 3 — Fraud / abuse signals

Agent 3 detects **suspicious or low-signal** reviews using **custom Python tools** (rules + simple statistics). Optional **Ollama** can summarize risks using **only** the structured `fraud_summary` + a small sample of flagged rows.

---

## Tools

File: `src/review_mas/tools/fraud_tools.py`

| Function | Role |
|----------|------|
| `normalize_review_text(text)` | Lowercase, strip HTML tags, collapse whitespace (duplicate detection) |
| `assess_fraud_patterns(reviews)` | Returns `(fraud_assessments, fraud_summary)` |

Signals include (examples):

- `duplicate_review_text` — same normalized text appears more than once
- `very_short_review` — extremely little text
- `high_rating_very_short_text` — 5★ but suspiciously little text
- `low_rating_very_short_text` — 1★ with very little text
- `mostly_non_letters` — text dominated by symbols/numbers

---

## State outputs

- `fraud_assessments`: per-review scores + reasons + `flagged`
- `fraud_summary`: aggregates (`flagged_count`, `reason_counts`, `avg_fraud_score`, …)
- Optional: `fraud_summary["llm_fraud_overview"]` when `--use-ollama` is enabled

---

## Run only Agents 1–3 (write JSON)

```powershell
python scripts/run_agent3_fraud.py --dataset data/hf_reviews_electronics.csv --auto-product --out outputs/agent3_fraud.json
```

With Ollama overview:

```powershell
python scripts/run_agent3_fraud.py --dataset data/hf_reviews_electronics.csv --auto-product --use-ollama --ollama-model phi3 --out outputs/agent3_fraud_ollama.json
```

---

## Tests

```powershell
python -m pytest tests/test_fraud_agent.py -v
```
