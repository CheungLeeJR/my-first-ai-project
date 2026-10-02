# DSAI Research Project Portfolio

A research-oriented portfolio of four Data Science, AI, machine-learning, and algorithmic projects. The repository is organized to make the **research question, experimental design, evidence boundary, reproducibility path, and limitations** explicit rather than presenting code alone.

## Projects

| Project | Research framing | Current evidence | Status |
|---|---|---|---|
| [HKMU AI Study Assistant](./hkmu-ai-study-assistant) | Retrieval-augmented generation for multi-document study support | Deterministic system tests; evaluation harness added for retrieval/citation metrics | Research prototype |
| [EV Purchase Prediction](./ev-purchase-prediction-ensemble) | Leakage-safe tabular prediction and ensemble selection | 5-fold OOF design; earlier verified public score 0.94156; V2 result not yet re-verified | Reproducible competition study |
| [Kaggriculture Tournament Agent](./kaggriculture-tournament-agent) | Heuristic policy design for sequential resource allocation | Policy invariants and safety tests; benchmark/ablation protocol defined | Algorithmic research prototype |
| [Caregiver Interest Model Reproduction](./caregiver-interest-model-reproduction) | Imbalanced binary classification reproduction | 5-fold aggregate metrics comparing unbalanced vs fold-local undersampling | Research reproduction |

## Shared research standard

Every project now documents:

1. **Research question / hypothesis** — what is being tested and what is not.
2. **Methodology** — data flow, model/policy design, and leakage controls.
3. **Evaluation protocol** — primary metrics, baselines, ablations, and failure criteria.
4. **Reproducibility** — deterministic seeds where applicable, environment/dependency records, tests, and run artifacts.
5. **Evidence boundary** — measured results are separated from proposed experiments and future improvements.
6. **Limitations / responsible use** — no causal, clinical, or real-world claims are inferred from competition/reproduction results.
7. **Data governance** — secrets, restricted research data, row-level sensitive outputs, local databases, and generated artifacts are excluded from public Git history.

See [`RESEARCH_STANDARDS.md`](./RESEARCH_STANDARDS.md) for the portfolio-wide checklist.

## Reproduce / verify

The root GitHub Actions workflow runs the dependency-appropriate test suite for all four projects. Locally, enter a project directory and follow its `REPRODUCIBILITY.md`.

## Evidence policy

This portfolio intentionally does **not** convert unverified ideas into claims. In particular:

- the EV V2 pipeline is presented as a stronger experimental code path, not as a verified leaderboard improvement;
- the Kaggriculture agent has no fabricated tournament-score distribution;
- the RAG project has an evaluation protocol/harness but no invented retrieval-quality benchmark;
- the caregiver reproduction reports aggregate results only and excludes participant-level records.
