# Model Card — Caregiver Service-Interest Classifier

## Scope

Research reproduction of a binary classifier for caregiver service interest. The supplied project artifacts support a comparison between original imbalanced training folds and 1:1 random undersampling performed inside each training fold.

## Evaluation

Five-fold stratified cross-validation. Validation folds retain the original class distribution. Reported metrics are aggregate only.

| Strategy | Accuracy | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Unbalanced | 0.8060 | 0.7663 | 0.7338 | 0.4329 | 0.5432 |
| Balanced | 0.7327 | 0.7696 | 0.4982 | 0.6690 | 0.5707 |

These results show a trade-off: undersampling increases recall while lowering accuracy and precision. The appropriate operating point depends on the study objective and the consequences of false positives/false negatives.

## Data governance

The repository must not contain participant-level source data, rebuilt feature rows, addresses, free-text responses, or OOF rows unless the data owner explicitly authorizes that disclosure. The CLI therefore suppresses row-level outputs by default; they require the explicit `--include-row-level-artifacts` flag.

## Limitations

- The supplied artifact supports reproduction of the stated evaluation, not external validity.
- Aggregate performance should not be interpreted as evidence that the model is appropriate for individual care decisions.
- Public release of feature names, cohort details, code, and aggregate results remains subject to research-owner approval.
