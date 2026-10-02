# Research Protocol — EV Purchase Prediction

## Research objective

Evaluate whether leakage-safe feature engineering and model diversity improve out-of-fold discrimination for EV purchase prediction relative to simpler single-model baselines.

## Research questions

- **RQ1:** does nested, cross-fitted target encoding improve OOF ROC AUC over native categorical handling alone?
- **RQ2:** does CatBoost add useful predictive diversity to LightGBM models?
- **RQ3:** do OOF-selected probability/rank blends improve performance without using leaderboard feedback for weight selection?
- **RQ4:** are gains stable across folds rather than driven by a single validation split?

## Experimental design

- Primary evaluation: 5-fold stratified OOF ROC AUC.
- All learned preprocessing is fitted inside each outer training fold.
- Target encoding for outer-training rows is itself cross-fitted using inner folds.
- External leaderboard feedback is not used to select blend weights.
- The final competition submission is treated as an external check, not the model-selection objective.

## Baselines / ablations

1. LightGBM with native categorical processing.
2. LightGBM + leakage-safe fold target encoding.
3. CatBoost.
4. Pairwise probability blends.
5. Rank blends.
6. Three-model simplex blends when all models are available.

## Primary metric

ROC AUC, reported per fold and on complete OOF predictions.

## Recommended secondary analyses for the next rerun

- fold mean ± standard deviation;
- repeated stratified CV across several seeds;
- calibration/Brier score;
- feature-importance stability or SHAP diagnostics;
- error slices by major demographic/domain groups if allowed by the competition data.

## Evidence boundary

The supplied project history verifies a public leaderboard score of **0.94156** for the earlier stable pipeline. It does not contain a verified leaderboard result for the V2 pipeline, so V2 is treated as an experimental candidate until rerun.

This is a predictive competition study, not a causal study of EV adoption.
