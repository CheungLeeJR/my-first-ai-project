# Reproducibility — Kaggriculture Tournament Agent

## Runtime

The tournament agent itself uses only the Python standard library.

## Verify policy invariants

```bash
python -m pip install pytest
python -m pytest -q
python scripts/inspect_policy.py
```

## Determinism

Given the same observation object, the policy contains no stochastic decision step and should return the same action. Market/opponent state can still change the decision because they are part of the observation.

## What is reproducible today

- opening crop-map counts;
- disjoint long-term role allocation;
- animal-site exclusions;
- market-price monotonicity around inventory changes;
- safe-sell quantity bounds;
- legal fallback action shape;
- declared policy snapshot through `scripts/inspect_policy.py`.

## What still requires the original environment

Tournament-score benchmarking, opponent comparisons, and ablations require either the official environment or a faithful replay/simulator harness. Those results should be saved by seed/opponent and should not be inferred from unit tests.
