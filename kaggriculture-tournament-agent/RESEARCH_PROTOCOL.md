# Research Protocol — Kaggriculture Tournament Agent

## Research objective

Study a deterministic heuristic policy for sequential resource allocation under spatial, labor, inventory, production-cycle, and market constraints.

## Research questions

- **RQ1:** which parts of the policy contribute most to final tournament value: crop layout, livestock mix, worker scheduling, logistics, or market timing?
- **RQ2:** does zone-sticky distance-aware task assignment reduce wasted movement relative to purely greedy task priority?
- **RQ3:** how sensitive is the policy to opponent production pressure and market timing rules?
- **RQ4:** which resource bottlenecks dominate failure modes (labor, feed, inventory capacity, cash, or travel distance)?

## Experimental unit

A complete 30-day / 720-step episode against a specified opponent/environment seed.

## Primary outcome for a future benchmark

Final tournament score/value, reported as a distribution across seeds/opponents rather than a single best episode.

## Secondary outcomes

- animal-loss/escape count;
- missed watering/feed events;
- worker idle-action rate;
- travel actions per productive action;
- shed overflow events;
- realized average sale price by product;
- end-of-game unsold inventory.

## Required baselines / ablations

1. Full current policy.
2. No opponent-aware market/portfolio signals.
3. No zone stickiness in worker assignment.
4. Fixed crop mix without demand-flex tiles.
5. Fixed livestock mix.
6. Immediate selling vs demand-timed selling.
7. Reduced worker-hiring schedule.

## Statistical reporting

When a simulator/replay harness is available, run matched seeds across variants and report mean, median, standard deviation, and paired per-seed differences. Do not select only favorable episodes.

## Current evidence boundary

The repository currently proves policy invariants and safety properties with tests. It does **not** contain enough replay/simulator evidence to claim that the policy is optimal or that any individual heuristic improves tournament score.
