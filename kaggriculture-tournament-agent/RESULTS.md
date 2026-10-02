# Results and Evidence Ledger

## Verified engineering evidence

The current test suite validates seven policy/safety invariants, and the policy-inspection script exposes the declared layout and market-curve behavior without requiring the tournament environment.

## Not yet established

No statistically meaningful episode benchmark is present in the supplied artifacts. Therefore this repository does not claim:

- optimality;
- a specific tournament ranking;
- a score improvement attributable to any one heuristic;
- robustness across opponent policies or random seeds.

## Next research milestone

Build or recover a replay/simulation harness, then run the ablation matrix in `RESEARCH_PROTOCOL.md` on matched seeds. Save episode-level metrics locally and publish aggregate tables/plots when permitted.
