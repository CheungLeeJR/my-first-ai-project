# Reproducibility — HKMU AI Study Assistant

## Environment

Recommended Python: **3.11**. Install dependencies with `pip install -r requirements.txt` and create `.env` from `.env.example`.

## Deterministic verification

```bash
python -m pytest -q
```

Core tests and fake-provider system tests avoid live OpenAI calls. Live-provider behavior is intentionally outside the deterministic test boundary.

## Offline research evaluation

The repository includes `scripts/evaluate_retrieval.py`, which evaluates precomputed retrieval/citation outputs from JSONL without requiring an API key.

```bash
python scripts/evaluate_retrieval.py path/to/eval.jsonl
```

Each JSONL record can contain:

```json
{"query":"...","relevant_chunk_ids":["c1"],"retrieved_chunk_ids":["c2","c1"],"expected_citation_ids":["docA:p3"],"predicted_citation_ids":["docA:p3"]}
```

The script reports aggregate Recall@k, Hit Rate@k, MRR, and citation precision/recall/F1.

## Artifacts not committed

`.env`, API keys, uploaded PDFs, local SQLite databases, FAISS indexes, logs, and generated evaluation rows are excluded from version control.

## Reproduction boundary

Full live reproduction depends on provider/model availability and a valid API key. The research-facing offline evaluator and deterministic tests are designed so a reviewer can inspect the evaluation logic without provider access.
