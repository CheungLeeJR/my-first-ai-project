from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import TARGET, load_authorized_dataset, run_cv, summarize


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the caregiver-interest CatBoost evaluation from an authorized raw CSV.")
    parser.add_argument("raw_csv", type=Path, help="Path to authorized raw data. The file is never copied into the repository.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated aggregate artifacts.")
    parser.add_argument(
        "--include-row-level-artifacts",
        action="store_true",
        help="Explicitly write rebuilt participant rows and OOF predictions. Off by default to reduce accidental disclosure risk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    frame, audit = load_authorized_dataset(args.raw_csv)
    y = frame[TARGET].astype(int).to_numpy()

    balanced_folds, balanced_audit, balanced_prob, balanced_pred, balanced_cm = run_cv(frame, "Balanced")
    unbalanced_folds, _, unbalanced_prob, unbalanced_pred, unbalanced_cm = run_cv(frame, "Unbalanced")
    folds = pd.concat([unbalanced_folds, balanced_folds], ignore_index=True)

    pd.DataFrame([audit]).to_csv(output / "00_Raw_Cleaning_Audit.csv", index=False)
    folds.to_csv(output / "01_5Fold_Metrics.csv", index=False)
    balanced_audit.to_csv(output / "02_Fold_Balance_Audit.csv", index=False)
    summary = pd.DataFrame([
        summarize("Unbalanced", unbalanced_folds),
        summarize("Balanced", balanced_folds),
    ])
    summary.to_csv(output / "03_Summary_Metrics.csv", index=False)
    pd.DataFrame(balanced_cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(output / "04_Confusion_Matrix_Balanced.csv")
    pd.DataFrame(unbalanced_cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(output / "05_Confusion_Matrix_Unbalanced.csv")

    if args.include_row_level_artifacts:
        frame.to_csv(output / "00_Rebuilt_Model_Dataset_From_Raw.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "y_true": y,
                "prob_unbalanced": unbalanced_prob,
                "pred_unbalanced": unbalanced_pred,
                "prob_balanced": balanced_prob,
                "pred_balanced": balanced_pred,
            }
        ).to_csv(output / "06_OOF_Predictions.csv", index=False)

    config = {
        "raw_file": "[local authorized input; path intentionally not persisted]",
        "final_rows": len(frame),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "balancing": "Random undersampling in training fold only, 1:1",
        "validation": "unchanged original class distribution",
        "model": "CatBoost All35",
        "threshold": 0.5,
        "row_level_artifacts_written": bool(args.include_row_level_artifacts),
        "note": "Numeric missing values are handled natively by CatBoost. No validation-fold statistics are used for preprocessing.",
    }
    (output / "07_Run_Config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    if not args.include_row_level_artifacts:
        print("\nPrivacy guard: row-level rebuilt data and OOF predictions were NOT written.")


if __name__ == "__main__":
    main()
