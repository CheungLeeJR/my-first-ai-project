# DSAI Project Portfolio

A curated collection of Data Science, AI, machine-learning, and algorithmic projects. The repositories were cleaned for reproducibility, testability, and public review rather than uploaded as raw coursework folders.

## Featured projects

| Project | Focus | Key engineering / ML ideas | Verification |
|---|---|---|---|
| [HKMU AI Study Assistant](./hkmu-ai-study-assistant) | RAG / AI engineering | Multi-PDF retrieval, FAISS, SQLite, OCR fallback, citations, deterministic fake-provider tests | 3 local tests passed |
| [EV Purchase Prediction](./ev-purchase-prediction-ensemble) | Tabular ML / Kaggle | 5-fold OOF, leakage-safe preprocessing, nested target encoding, LightGBM/CatBoost ensemble | 3 local tests passed; earlier verified public score 0.94156 |
| [Kaggriculture Tournament Agent](./kaggriculture-tournament-agent) | Agent / optimization | Task scheduling, logistics, resource allocation, market timing, opponent-aware policy | 6 local tests passed |
| [Caregiver Interest Model Reproduction](./caregiver-interest-model-reproduction) | Research reproduction / imbalanced classification | Fold-local undersampling, CatBoost CV, aggregate evaluation, privacy-by-default outputs | 4 local tests passed |

## Portfolio-wide quality controls

- Public-facing README and reproducibility instructions for every project
- Unit tests that avoid requiring private competition/research data where possible
- Root GitHub Actions workflow covering all four projects
- `.gitignore` rules for generated artifacts, secrets, local databases, and row-level outputs
- Explicit evidence boundaries: no leaderboard/model-performance claims beyond supplied artifacts
- Restricted caregiver participant-level data intentionally excluded from the public repository

## Project map

```text
.
├── hkmu-ai-study-assistant/
├── ev-purchase-prediction-ensemble/
├── kaggriculture-tournament-agent/
├── caregiver-interest-model-reproduction/
└── .github/workflows/portfolio-ci.yml
```

## Notes

The caregiver project is published as a **sanitized code + aggregate-results reproduction**. The raw research dataset, rebuilt participant-level dataset, and row-level OOF predictions are not included.

For the EV project, the supplied history supports a public score of **0.94156** for the earlier stable pipeline. The stronger V2 code is included, but no unsupported leaderboard score is claimed for it.
