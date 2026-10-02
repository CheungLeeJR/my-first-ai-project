# Model Card

## Task

Binary classification of whether a customer will purchase an electric vehicle in the attached Kaggle competition dataset.

## Validation design

- 5-fold stratified out-of-fold evaluation.
- Preprocessing statistics are fitted only on each outer training fold.
- Target encoding is nested/cross-fitted inside the outer training fold.
- Ensemble weights are selected from OOF predictions only.

## Reported evidence

The uploaded project history contains a verified public leaderboard score of **0.94156** for the earlier stable pipeline. The stronger V2 pipeline is included as an experiment, but the supplied files do not contain a verified leaderboard score for that version. It should therefore be treated as an unverified improvement candidate until rerun.

## Intended use

Portfolio and competition experimentation. This model should not be used for real lending, insurance, employment, pricing, or other high-impact decisions.

## Limitations

- Competition data may contain synthetic patterns that do not transfer to real EV adoption behavior.
- Public leaderboard performance is not evidence of real-world calibration or causal validity.
- Feature engineering is competition-specific and intentionally includes pattern-oriented transforms.
