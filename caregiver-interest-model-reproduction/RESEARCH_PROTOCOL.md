# Research Protocol — Caregiver Service-Interest Reproduction

## Research objective

Reproduce and audit a binary classification workflow for caregiver service interest, with special attention to class imbalance and leakage-safe evaluation.

## Research questions

- **RQ1:** how does 1:1 random undersampling inside the training fold change recall, precision, F1, and ROC AUC relative to training on the original class distribution?
- **RQ2:** is the change consistent across the five stratified validation folds?
- **RQ3:** what operating trade-off is introduced by using the same 0.5 threshold after balancing?

## Experimental design

- 5-fold `StratifiedKFold` by default.
- Validation folds retain the original class distribution.
- For the balanced strategy, undersampling is performed **only on the training indices of the current fold**.
- The same feature construction and CatBoost family are used for both strategies.
- Default threshold: 0.5.

## Primary metrics

ROC AUC, precision, recall, and F1. Accuracy is reported but should not dominate interpretation under class imbalance.

## Recommended extension metrics for the next authorized rerun

- precision-recall AUC / average precision;
- balanced accuracy;
- Brier score and calibration plot;
- threshold-performance curves;
- bootstrap or repeated-CV uncertainty analysis if justified by the study design.

## Interpretation

The current aggregate reproduction shows a recall/precision trade-off under undersampling. It does not establish that either strategy is universally preferable; the appropriate operating point depends on the study objective and error costs.

## Data/ethics boundary

This repository is not evidence for individual care decisions. Participant-level data remain restricted, and any public release remains subject to authorization from the relevant data owner/supervisor.
