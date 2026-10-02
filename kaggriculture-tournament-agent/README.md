# Kaggriculture Tournament Agent

A standard-library-only strategy agent for a 30-day / 720-step farming tournament environment. It combines heuristic planning, multi-worker task assignment, production scheduling, logistics, market timing, and opponent-aware rules in one portable submission module.

## Strategy overview

The agent is built around five interacting systems:

1. **Farm layout** — a dense opening followed by three-quadrant expansion, with crop roles assigned by distance and production cycle.
2. **Livestock portfolio** — staged cow/sheep purchases with near-shed pasture placement and feed-runway constraints.
3. **Worker scheduler** — priority-based task generation plus distance-aware, zone-sticky assignment across farmer + hired hands.
4. **Logistics** — feed/fertilizer pickup, harvest return, shed capacity management, and emergency animal-survival handling.
5. **Market execution** — demand-timed selling, reserve management, dynamic price-curve checks, and terminal liquidation.

## Selected design choices

- Opening crop mix: **9 melon / 10 wheat / 2 carrot**
- Three-quadrant expansion rather than buying all land
- **14** near-shed pasture sites
- Long-term role split: **38 strawberry / 8 melon / 7 feed-wheat / 8 flex** sites
- Opponent-visible production as an additional portfolio/sell-timing signal
- Final-day harvest, return, and liquidation safeguards
- Safe fallback to legal `PASS` actions if an unexpected observation causes an exception

## Repository layout

```text
.
├── src/agent.py                 # self-contained tournament submission
├── tests/test_agent.py          # policy invariants + safety tests
├── scripts/inspect_policy.py    # inspect layout and market curves offline
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Entry point

The tournament expects:

```python
from agent import agent
result = agent(obs)
```

`agent(obs)` returns farmer action, hired-hand actions, and market orders.

## Inspect the policy without the tournament environment

```bash
python scripts/inspect_policy.py
```

This prints the declared opening/long-term site counts and sample market prices around equilibrium inventory.

## Test

The runtime agent has no third-party dependencies. Tests only require pytest:

```bash
python -m pytest
```

The current suite checks fallback action shape, opening/long-term layout invariants, market monotonicity, and sell-quantity safety.

## Engineering note

The strategy remains one self-contained `agent.py` because tournament platforms often impose single-file or restricted-packaging constraints. Documentation, tests, and inspection tooling are kept outside the submission module.

## Next improvements

- Add replay-based regression tests against known tournament states
- Build a lightweight local simulator/trace harness
- Replace hand-tuned task priorities with offline search or bandit tuning
- Record strategy variants and score distributions across seeds/opponents
