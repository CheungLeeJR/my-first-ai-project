# Caregiver Service-Interest Classification — Reproduction Pipeline

A privacy-conscious research reproduction project for caregiver service-interest classification. The public repository contains the modelling pipeline, tests, and **aggregate** evaluation outputs only; restricted participant-level source data and row-level predictions are excluded.

## What this project demonstrates

- Reproducible cleaning and feature construction from survey-style tabular data
- Explicit missing-value and sentinel-value handling
- CatBoost binary classification with **5-fold StratifiedKFold**
- A fair comparison between original imbalanced training and **1:1 undersampling applied inside training folds only**
- Validation on the original class distribution
- Accuracy, ROC AUC, precision, recall, F1, confusion matrices, and fold-balance audits
- Privacy guardrails that disable participant-level outputs by default
- Dependency-light unit tests that do not require the restricted dataset

## Aggregate reproduced results

| Strategy | Accuracy | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Unbalanced | 0.8060 | 0.7663 | 0.7338 | 0.4329 | 0.5432 |
| Balanced | 0.7327 | 0.7696 | 0.4982 | 0.6690 | 0.5707 |

The undersampled strategy increases recall substantially while reducing accuracy and precision. This is presented as an operating trade-off rather than a universal winner; see `MODEL_CARD.md` for interpretation boundaries.

## Repository layout

```text
.
├── src/
│   ├── pipeline.py        # cleaning, features, CV and evaluation helpers
│   └── run_pipeline.py    # CLI reproduction entry point
├── tests/test_pipeline.py
├── results/               # aggregate-only reproduced outputs
├── MODEL_CARD.md
├── PUBLICATION_REVIEW.md
└── requirements.txt
```

## Reproduce aggregate outputs

Use only a dataset you are authorized to access:

```bash
pip install -r requirements.txt
python src/run_pipeline.py /path/to/authorized_raw.csv --output-dir outputs
```

By default the command writes only aggregate audits, fold metrics, summary metrics, confusion matrices, and a sanitized run configuration.

Participant-level artifacts require an explicit opt-in:

```bash
python src/run_pipeline.py /path/to/authorized_raw.csv \
  --output-dir outputs \
  --include-row-level-artifacts
```

Do **not** use that flag for public outputs unless the data owner has explicitly authorized disclosure.

## Test without research data

```bash
pip install numpy pandas scikit-learn pytest
python -m pytest
```

The tests cover date/age parsing, missing-value behavior, item aggregation, and fold-local 1:1 undersampling without using participant records.

## Public-release boundary

Included in the public portfolio:

- modelling and feature-building code;
- aggregate cleaning counts;
- fold-level and summary metrics;
- aggregate confusion matrices;
- sanitized run configuration;
- publication/data-governance notes.

Excluded:

- raw survey data;
- the 954-row rebuilt participant-level feature table;
- the 954-row OOF prediction file;
- names, contact details, addresses, free text, or other direct identifiers.

The public code package is intentionally designed so the restricted dataset is **not required to inspect or test the core engineering work**.

## Research preparation

- Research questions and imbalance experiment design: [`RESEARCH_PROTOCOL.md`](./RESEARCH_PROTOCOL.md)
- Seed/fold/threshold controls: [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)
- Aggregate result interpretation: [`RESULTS.md`](./RESULTS.md)

The public release remains aggregate-only and does not claim external validity or individual decision suitability.
