# Results and Evidence Ledger

## Verified historical evidence

- Earlier stable pipeline: **public leaderboard score 0.94156** (supported by the uploaded project history).

## V2 experimental pipeline

The repository contains a stronger V2 design with nested target encoding, LightGBM/CatBoost diversity, and OOF-only blend selection. The supplied artifacts do **not** contain a verified V2 leaderboard result, so no improvement over 0.94156 is claimed here.

## What should be recorded on the next full rerun

| Item | Required evidence |
|---|---|
| OOF model comparison | per-fold AUC, OOF AUC, fold SD |
| Blend selection | OOF-only weights / ranking |
| Reproducibility | `run_metadata.json` |
| Submission integrity | schema, row count, unique ID checks |
| External result | leaderboard score + submission timestamp/screenshot if appropriate |

## Interpretation boundary

A competition score measures performance on the competition distribution. It is not evidence of causal EV-purchase drivers, real-market calibration, or deployment suitability.
