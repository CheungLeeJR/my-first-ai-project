# Research Portfolio Standards

The featured projects in this portfolio use the following standard for research preparation and review.

## 1. Research question first

Each project should state a concrete question that can be evaluated rather than describing only a software feature.

## 2. Testable hypotheses

Hypotheses are labelled as hypotheses until supported by experiment results. Planned improvements are not written as measured gains.

## 3. Explicit baselines and ablations

Where applicable, projects identify simpler baselines and controlled ablations so the effect of individual design choices can be studied.

## 4. Leakage-aware evaluation

Machine-learning evaluation keeps learned preprocessing inside the training fold. Competition leaderboard feedback is treated as an external check rather than a substitute for validation design.

## 5. Reproducibility

Projects should record the relevant random seed, fold configuration, package/runtime versions, input fingerprints or data version, and generated experiment metadata where practical.

## 6. Evidence ledger

Verified historical results, current reproducible measurements, and future experiments are clearly separated. Unsupported performance claims are avoided.

## 7. Tests are not research results

Unit and integration tests establish software invariants and implementation correctness. They do not by themselves establish empirical model quality or superiority.

## 8. Limitations and external validity

Each research-facing project describes where conclusions do not generalize—for example from synthetic competition data to real deployment or from a local RAG benchmark to broad educational effectiveness.

## 9. Data governance

Raw restricted data, participant-level derived datasets, direct identifiers, private documents, secrets, and row-level predictions are excluded from public repositories unless publication is explicitly authorized.

## 10. Reproducible next step

Future research work should be framed as an experiment that can be run, recorded, compared, and reviewed—not simply as a feature wishlist.
