from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
SEED = 42


@dataclass(frozen=True)
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    features: list[str]
    target: np.ndarray


def encode_target(series: pd.Series) -> np.ndarray:
    """Normalize the competition target to an integer 0/1 array."""
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        if mapped.isna().any():
            bad = sorted(series[mapped.isna()].astype(str).unique().tolist())[:5]
            raise ValueError(f"Unsupported target labels: {bad}")
        return mapped.astype("int8").to_numpy()

    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not set(values.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary 0/1 or yes/no labels.")
    return values.astype("int8").to_numpy()


def validate_bundle(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    target: str = TARGET,
    id_column: str = ID_COLUMN,
) -> DataBundle:
    """Fail fast on schema, ordering, duplicate-ID, and target issues."""
    if target not in train.columns:
        raise ValueError(f"Missing target column: {target}")
    if id_column not in train.columns or id_column not in test.columns:
        raise ValueError(f"Missing ID column: {id_column}")

    features = [c for c in train.columns if c not in (target, id_column)]
    test_features = [c for c in test.columns if c != id_column]
    if features != test_features:
        raise ValueError("Train/test feature columns or order do not match.")
    if list(sample_submission.columns) != [id_column, target]:
        raise ValueError("Submission columns must be [id, target].")
    if len(sample_submission) != len(test):
        raise ValueError("Submission row count does not match test rows.")
    if not sample_submission[id_column].reset_index(drop=True).equals(test[id_column].reset_index(drop=True)):
        raise ValueError("Submission IDs are not in the same order as test IDs.")
    if not train[id_column].is_unique or not test[id_column].is_unique:
        raise ValueError("Duplicate IDs detected.")

    y = encode_target(train[target])
    return DataBundle(train, test, sample_submission, features, y)


def engineer(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Competition-specific feature engineering that never uses labels."""
    d = df[features].copy()
    inc, com, age = "Annual_Income_USD", "Daily_Commute_km", "Age"
    home, work = "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"

    if inc in d:
        d["income_log"] = np.log1p(d[inc].clip(lower=0))
        d["income_k"] = (d[inc] // 1000).astype("int32")
        d["income_mod1000"] = (d[inc] % 1000).astype("int16")
        d["income_floor30000"] = (d[inc] == 30000).astype("int8")
        for quantum in [100, 250, 500, 1000, 2500, 5000, 10000]:
            d[f"income_round_{quantum}"] = (d[inc] / quantum).round().astype("int32")

    if com in d:
        d["commute_log"] = np.log1p(d[com].clip(lower=0))
        d["commute_round1"] = d[com].round().astype("int16")
        d["commute_x10_round"] = (d[com] * 10).round().astype("int16")
        d["commute_floor5"] = (d[com] == 5).astype("int8")
        for quantum in [1, 2, 5, 10]:
            d[f"commute_bin_{quantum}"] = (d[com] // quantum).astype("int16")

    if inc in d and age in d:
        d["income_per_age"] = d[inc] / (d[age] + 1)

    if home in d and work in d:
        d["charging_total"] = d[home] + d[work]
        d["charging_min"] = d[[home, work]].min(axis=1)
        d["charging_max"] = d[[home, work]].max(axis=1)
        d["charging_gap"] = d[work] - d[home]
        d["charging_ratio"] = (d[work] + 1) / (d[home] + 1)

    if com in d and home in d:
        d["commute_per_home_station"] = d[com] / (d[home] + 1)
    if com in d and work in d:
        d["commute_per_work_station"] = d[com] / (d[work] + 1)

    combos = [
        ["Subsidy_Available", "Environmental_Concern_Level"],
        ["Home_Charging_Possible", "Range_Anxiety_Level"],
        ["City_Type", "Range_Anxiety_Level"],
        ["Current_Car_Type", "City_Type"],
        ["Subsidy_Available", "Home_Charging_Possible"],
        ["Environmental_Concern_Level", "Range_Anxiety_Level"],
    ]
    for columns in combos:
        if all(c in d for c in columns):
            d["combo_" + "_".join(columns)] = d[columns].astype(str).agg("|".join, axis=1)
    return d


def cross_fitted_target_encode(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y_train: np.ndarray,
    columns: list[str],
    *,
    smoothing: float = 40.0,
    inner_folds: int = 5,
    seed: int = SEED + 917,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe nested target encoding for an outer CV training fold."""
    a, b, t = train_x.copy(), valid_x.copy(), test_x.copy()
    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Target encoding requires both classes in the training fold.")

    splitter = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for column in columns:
        key_a = a[column].fillna("__MISSING__").astype(str).reset_index(drop=True)
        key_b = b[column].fillna("__MISSING__").astype(str)
        key_t = t[column].fillna("__MISSING__").astype(str)
        encoded_train = np.zeros(len(a), dtype="float32")

        for inner_train, inner_valid in splitter.split(np.zeros(len(y_train)), y_train):
            prior = float(y_train[inner_train].mean())
            stats = pd.DataFrame(
                {"key": key_a.iloc[inner_train].to_numpy(), "y": y_train[inner_train]}
            ).groupby("key")["y"].agg(["mean", "count"])
            mapping = ((stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)).to_dict()
            encoded_train[inner_valid] = key_a.iloc[inner_valid].map(mapping).fillna(prior).to_numpy(dtype="float32")

        prior = float(y_train.mean())
        stats = pd.DataFrame({"key": key_a.to_numpy(), "y": y_train}).groupby("key")["y"].agg(["mean", "count"])
        mapping = ((stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)).to_dict()
        a[f"{column}_te"] = encoded_train
        b[f"{column}_te"] = key_b.map(mapping).fillna(prior).to_numpy(dtype="float32")
        t[f"{column}_te"] = key_t.map(mapping).fillna(prior).to_numpy(dtype="float32")

    return a, b, t


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str = TARGET,
    id_column: str = ID_COLUMN,
) -> None:
    """Hard validation used immediately before a competition submission is written."""
    errors: list[str] = []
    if submission.shape != sample_submission.shape:
        errors.append("shape")
    if submission.columns.tolist() != sample_submission.columns.tolist():
        errors.append("columns")
    if id_column in submission and not submission[id_column].reset_index(drop=True).equals(test[id_column].reset_index(drop=True)):
        errors.append("id/order")
    if target in submission:
        pred = pd.to_numeric(submission[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(pred).all():
            errors.append("nan/inf")
        if not ((pred >= 0) & (pred <= 1)).all():
            errors.append("range")
        if len(pred) and (np.std(pred) <= 0 or len(np.unique(pred)) < min(100, len(pred))):
            errors.append("constant/unique")
    if any(str(c).startswith("Unnamed") for c in submission.columns):
        errors.append("index column")
    if errors:
        raise ValueError("Submission validation failed: " + ", ".join(errors))


def locate_competition_data(root: Path) -> Path:
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    candidates: list[tuple[int, Path]] = []
    for train_path in root.rglob("train.csv"):
        directory = train_path.parent
        if all((directory / name).is_file() for name in required):
            score = 100 if "playground-series-s6e9" in str(directory).lower() else 0
            if (directory / "train.csv").stat().st_size > (directory / "test.csv").stat().st_size:
                score += 10
            candidates.append((score, directory))
    if not candidates:
        raise FileNotFoundError("Could not find train.csv, test.csv and sample_submission.csv under the input root.")
    return sorted(candidates, key=lambda item: (item[0], str(item[1])), reverse=True)[0][1]
