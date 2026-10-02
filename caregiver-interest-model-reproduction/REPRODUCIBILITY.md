# Reproducibility — Caregiver Interest Model

## Environment

Recommended Python: **3.11**.

```bash
pip install -r requirements.txt
```

## Authorized-data run

```bash
python src/run_pipeline.py /path/to/authorized_raw.csv --output-dir outputs
```

The default is privacy-preserving: row-level rebuilt data and OOF predictions are not written.

## Deterministic controls

The research-ready local version supports explicit random seed, fold count, and threshold controls. Keep those settings fixed and record them for each experiment.

## Row-level research artifacts

`--include-row-level-artifacts` is an explicit local-only opt-in. Do not publish those outputs unless disclosure is authorized.

## Tests

```bash
python -m pytest -q
```

Tests use synthetic values and do not require participant data.

## Reproduction boundary

The public `results/` directory contains aggregate outputs from the supplied reproduction artifact. A future rerun with changed seeds/folds/metrics should be archived as a separate experiment rather than silently replacing the evidence ledger.
