# Results and Interpretation

## Aggregate reproduced results

| Strategy | Accuracy | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Unbalanced | 0.8060 | 0.7663 | 0.7338 | 0.4329 | 0.5432 |
| Balanced | 0.7327 | 0.7696 | 0.4982 | 0.6690 | 0.5707 |

## Observed trade-off

Relative to unbalanced training, fold-local 1:1 undersampling increases recall and F1 in the supplied aggregate results, while reducing accuracy and precision. ROC AUC is similar between the two strategies.

This should be read as an **operating trade-off**, not as a universal ranking. A study that prioritizes identifying more interested caregivers may value recall differently from one where false positives are costly.

## Evidence limitations

- Results are internal cross-validation estimates on the supplied cohort.
- There is no external validation cohort in the supplied artifacts.
- A fixed 0.5 threshold is not necessarily decision-optimal after undersampling.
- No causal interpretation is supported.
- The public repository does not contain participant-level data required to independently reconstruct the cohort.

## Next authorized experiment

On the next rerun, add precision-recall AUC, calibration, threshold curves, and repeated-CV or bootstrap uncertainty where methodologically appropriate.
