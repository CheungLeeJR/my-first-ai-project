# Portfolio Research Standards

This checklist is used across the portfolio to keep projects suitable for supervisor review, research applications, and later conversion into formal experiments.

## 1. Problem formulation

- State a concrete research question or engineering hypothesis.
- Separate prediction/association from causal claims.
- Define the intended scope and the non-goals.

## 2. Experimental design

- Identify baselines and meaningful ablations.
- Fit all learned preprocessing only on training partitions.
- Keep validation/test information outside model-selection steps.
- Use fixed, documented random seeds when stochastic procedures are involved.
- Report the primary metric before inspecting final external evaluation when possible.

## 3. Reproducibility

- Provide environment/dependency files and a single documented entry point.
- Record seeds, fold count, data dimensions, model configuration, and package versions in generated run metadata where practical.
- Keep tests independent of restricted datasets where possible.
- Preserve generated results separately from source code.

## 4. Result reporting

- Distinguish **measured evidence**, **interpretation**, and **future hypotheses**.
- Report variability across folds/seeds when available, not only a single best number.
- Avoid claiming generalization beyond the evaluation population/environment.
- Do not present leaderboard optimization as scientific external validity.

## 5. Data governance and responsible use

- Never commit credentials, raw restricted participant records, or unnecessary row-level research outputs.
- Document whether data are public, competition-provided, synthetic, local, or restricted.
- For human-participant or potentially sensitive data, publish only what is authorized by the data owner/supervisor.
- Do not imply clinical, admissions, lending, employment, or other high-impact suitability without appropriate validation and governance.

## 6. Research-readiness review before using a project in an application

A project should be able to answer, in a few minutes:

- What is the question?
- What is the baseline?
- What changed?
- How was leakage/confounding controlled?
- What metric tests the hypothesis?
- What evidence has actually been measured?
- What failed or remains uncertain?
- Can another person reproduce the result with authorized data?
