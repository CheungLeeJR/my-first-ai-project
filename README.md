# Data Science & AI Research Portfolio

A research-oriented portfolio of selected machine-learning, RAG, sequential-decision, and applied data-science projects. Each substantial project now lives in its own repository so its research question, implementation, evidence, tests, and reproducibility history can be reviewed independently.

## Featured research projects

### 1. HKMU AI Study Assistant
**Multi-document RAG · Retrieval evaluation · Citation traceability · OCR**

Persistent conversational RAG for course PDFs using FAISS, SQLite, OCR fallback, page-level citations, and deterministic tests. The research layer defines retrieval/citation hypotheses and an offline evaluator for Recall@k, Hit Rate@k, MRR, and citation precision/recall/F1.

➡️ https://github.com/CheungLeeJR/hkmu-ai-study-assistant

### 2. EV Purchase Prediction
**Leakage-safe ML · OOF validation · Nested target encoding · LightGBM/CatBoost**

Research-oriented competition pipeline focused on experimental validity: fold-local preprocessing, nested cross-fitted target encoding, OOF-only ensemble selection, reproducibility metadata, input fingerprints, and explicit evidence boundaries.

➡️ https://github.com/CheungLeeJR/ev-purchase-prediction

### 3. Kaggriculture Tournament Agent
**Sequential decision making · Resource allocation · Scheduling · Ablation design**

A deterministic standard-library tournament agent combining spatial planning, multi-worker scheduling, production logistics, market timing, and opponent-aware rules. The research protocol defines matched-seed ablations rather than claiming performance from unit tests alone.

➡️ https://github.com/CheungLeeJR/kaggriculture-tournament-agent

### 4. Caregiver Service-Interest Classification
**Imbalanced classification · Reproduction study · CatBoost · Data governance**

A privacy-conscious reproduction pipeline comparing original imbalanced training with fold-local 1:1 undersampling. The public repository contains code and aggregate evidence only; participant-level source data and row-level predictions are excluded.

➡️ https://github.com/CheungLeeJR/caregiver-interest-model-reproduction

## Other selected projects

- **AI Research Trend Mining** — https://github.com/CheungLeeJR/AI-Research-Trend-Mining
- **FoodBridge** — https://github.com/CheungLeeJR/foodbridge-project
- **Lost & Found Project** — https://github.com/CheungLeeJR/Lost_and_found_project

## Research standard

The featured repositories are structured around a common evidence chain:

**Research question → hypothesis → method → baseline/ablation → evaluation → reproducibility → evidence → limitations → data governance**

See [`RESEARCH_STANDARDS.md`](./RESEARCH_STANDARDS.md) for the portfolio-wide standard.

## Evidence policy

This portfolio distinguishes verified results from planned experiments. Competition scores are not presented as real-world validity, engineering tests are not presented as empirical research results, and restricted participant-level data are not published.
