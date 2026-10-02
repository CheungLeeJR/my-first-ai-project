# EV Purchase Prediction — Leakage-Safe Ensemble

A Kaggle-style binary classification project for EV purchase prediction using competition-aware feature engineering, 5-fold out-of-fold validation, LightGBM, optional CatBoost, nested target encoding, and OOF-only blend selection.

## What this project demonstrates

- Strict train/test/submission schema validation before modelling
- Competition-specific feature engineering without label access
- **5-fold Stratified OOF** evaluation
- Fold-local preprocessing to prevent validation leakage
- **Nested cross-fitted target encoding** inside each outer training fold
- LightGBM + CatBoost model diversity
- OOF-only probability/rank blending rather than leaderboard-tuned weights
- Hard validation of generated submission files
- Unit-tested, reusable validation/feature-engineering helpers

## Verified evidence from the uploaded project history

The earlier stable pipeline produced a valid public Kaggle submission with a **0.94156** public score. The stronger V2 pipeline is included as the preferred experimental pipeline, but the supplied artifact does **not** contain a verified leaderboard result for that stronger version. This repository therefore does not claim one.

See `MODEL_CARD.md` for the evidence boundary and limitations.

## Leakage-control design

For every outer CV fold:

1. validation rows are held out completely;
2. numeric imputation statistics come only from the outer-training partition;
3. categorical dictionaries are learned only from training data;
4. target encoding for training rows is generated with inner cross-fitting;
5. validation/test target encodings use mappings fitted only on outer-training labels;
6. ensemble weights are selected using genuine OOF AUC only.

## Repository layout

```text
.
├── notebooks/
│   ├── kaggle_ev_top10_strong_v2.ipynb   # preferred Kaggle notebook
│   └── kaggle_ev_run_all.ipynb            # earlier stable baseline
├── src/
│   ├── train.py                            # full strong-V2 competition runner
│   └── pipeline.py                         # tested schema/feature/TE helpers
├── tests/test_pipeline.py
├── artifacts/audit/                        # aggregate/non-row-level data audits
├── MODEL_CARD.md
└── requirements.txt
```

## Run on Kaggle

Attach the competition dataset containing `train.csv`, `test.csv`, and `sample_submission.csv`, then run:

```bash
python src/train.py
```

The competition runner defaults to `/kaggle/input` and `/kaggle/working`. For local use:

```bash
KAGGLE_INPUT_ROOT=/path/to/input OUTPUT_DIR=./outputs python src/train.py
```

## Test the reusable core

```bash
pip install numpy pandas scikit-learn pytest
python -m pytest
```

The tests use small synthetic frames and do not require the competition dataset.

## Outputs

The full competition pipeline writes:

- `submission_best.csv`
- `submission_safe.csv`
- model-specific submissions
- `oof_predictions.csv`
- `model_comparison.csv`
- `run_summary.txt`
- `run_metadata.json`

Generated submissions and row-level OOF files are excluded from Git by default.

## Why OOF matters

Leaderboard feedback is noisy and can encourage accidental overfitting. This project keeps model and blend selection inside cross-validation, using the public leaderboard only as an external final check.

## Next improvements

- Re-run strong V2 and record reproducible OOF + leaderboard results
- Add repeated-CV stability analysis and calibration diagnostics
- Add SHAP-based feature diagnostics and error slices
- Tune only inside CV (for example with constrained Optuna search)

## Portfolio verification

- Reusable pipeline tests: **5 passed** during portfolio cleanup.
- Generated submissions and row-level OOF predictions are ignored by Git.
- Performance reporting distinguishes the verified earlier public score (**0.94156**) from the stronger V2 code path, whose leaderboard result was not present in the supplied artifacts.

## Research preparation

- Experimental questions and ablations: [`RESEARCH_PROTOCOL.md`](./RESEARCH_PROTOCOL.md)
- Reproduction controls and run metadata: [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)
- Verified-vs-pending evidence ledger: [`RESULTS.md`](./RESULTS.md)

A full run now records seed, fold count, package versions, data dimensions, and SHA-256 input fingerprints in `run_metadata.json`.
