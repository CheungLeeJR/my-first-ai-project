# Reproducibility — EV Purchase Prediction

## Environment

Recommended Python: **3.11**.

```bash
pip install -r requirements.txt
```

## Data contract

The runner expects one directory containing `train.csv`, `test.csv`, and `sample_submission.csv`. The target is `Will_Buy_EV` and the identifier is `id`.

## Run

```bash
KAGGLE_INPUT_ROOT=/path/to/input OUTPUT_DIR=./outputs python src/train.py
```

Optional deterministic controls:

```bash
EXPERIMENT_SEED=42 CV_FOLDS=5 KAGGLE_INPUT_ROOT=/path/to/input OUTPUT_DIR=./outputs python src/train.py
```

## Generated research metadata

Each full run writes `run_metadata.json` containing random seed/fold count, Python/package versions, train/test dimensions, target rate, SHA-256 hashes of the three competition inputs, and model-availability information.

## Unit tests

```bash
python -m pytest -q
```

Tests use synthetic data and do not require competition files.

## Evidence discipline

Generated OOF rows and submissions are not committed by default. Report a new V2 leaderboard score only after a fresh run is archived with its run metadata, OOF model comparison, and submission validation output.
