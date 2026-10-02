from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

TARGET = "Interested(N/Y/noindicate)(0/1/999)"
DISTANCE = "address_centre_distance(meters)"
CG_DISEASE = [f"CG_Disease_{i}" for i in range(1, 15)]
CG_INCOME = [f"CG_IncomeSource_{i}" for i in range(1, 12)]
CG_SERVICE_USE = [f"CG_ServiceUse_{i}" for i in range(1, 8)]
CG_SERVICE_NEEDS = [f"CG_ServiceNeeds_{i}" for i in range(1, 9)]
MNST_GROUPS = {
    "Social_Support": ["MNST01", "MNST02", "MNST14"],
    "Mental_Health": ["MNST04", "MNST09", "MNST17", "MNST21"],
    "Caregiving_Ability": ["MNST05", "MNST11", "MNST13", "MNST16", "MNST19"],
    "Caregiving_Need": ["MNST06", "MNST07", "MNST12", "MNST20"],
    "Physical_Health": ["MNST08", "MNST10", "MNST22"],
}
MNST_ORPHANS = ["MNST03", "MNST15", "MNST18"]
BASE = [
    "CG_age", "CG_sex", "CG_house_group", "CG_married", "CG_disease_count",
    "CG_employ_recode", "CG_income_source_count", "CG_income_CSSA", "CG_district",
    "distance_km", "CR_count", "multiple_CR", "CG_other_service_heard",
    "CG_serviceuse_count", "CG_serviceneeds_count", "MNST_Social_Support_score",
    "MNST_Mental_Health_score", "MNST_Caregiving_Ability_score",
    "MNST_Caregiving_Need_score", "MNST_Physical_Health_score", "MNST03", "MNST15", "MNST18",
]
SERVICE = ["service_familiarity", "use_centre_based", "use_direct_care", "use_hotline", "use_other_type"]
WORK = ["work_context", "caregiving_blocks_work", "health_blocks_work"]
TEXT = ["other_applicable_count", "other_text_completed_count", "other_without_text_count", "other_text_completion_rate"]
FEATURES = BASE + SERVICE + WORK + TEXT
CATEGORICAL = {
    "CG_sex", "CG_house_group", "CG_married", "CG_employ_recode", "CG_income_CSSA",
    "CG_district", "multiple_CR", "CG_other_service_heard", "service_familiarity", "work_context",
}


def to_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def parse_age(value, reference_date: dt.date = dt.date(2025, 9, 26)):
    text = (value or "").strip()
    if not text:
        return np.nan
    dob = None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            dob = dt.datetime.strptime(text, fmt).date()
            break
        except ValueError:
            continue
    if dob is None:
        return np.nan
    age = reference_date.year - dob.year - ((reference_date.month, reference_date.day) < (dob.month, dob.day))
    return float(age) if 0 <= age <= 120 else np.nan


def category_value(row, column):
    value = (row.get(column, "") or "").strip()
    if not value:
        return np.nan
    number = to_float(value)
    if not np.isnan(number):
        return str(int(number)) if float(number).is_integer() else str(number)
    return value


def canonical_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        pass
    return text


def sum_items(row, columns, require_complete=True, fallback_zero_if=None):
    values = np.array([to_float(row.get(column, "")) for column in columns], float)
    if fallback_zero_if is not None:
        condition = to_float(row.get(fallback_zero_if, ""))
        if not np.isnan(condition) and condition == 0:
            return 0.0
    if require_complete and np.isnan(values).any():
        return np.nan
    if np.isnan(values).all():
        return np.nan
    return float(np.nansum(values))


def house_group(row):
    value = to_float(row.get("CG_HouseType", ""))
    if np.isnan(value):
        return np.nan
    code = int(value)
    if code in (1, 2):
        return "owned_or_subsidized_owned"
    if code in (0, 3, 4):
        return "public_or_rented"
    return "other"


def employment_group(row):
    value = to_float(row.get("CG_Employ", ""))
    return np.nan if np.isnan(value) else float(int(value) in (0, 1, 3))


def married(row):
    value = to_float(row.get("CG_Marital", ""))
    return np.nan if np.isnan(value) else float(int(value) == 1)


def build_features(row: dict) -> dict:
    """Rebuild the model feature row from one authorized raw survey record."""
    x = {
        "CG_age": parse_age(row.get("CG_DOB_intake", "")),
        "CG_sex": category_value(row, "CG_Sex"),
        "CG_house_group": house_group(row),
        "CG_married": married(row),
        "CG_disease_count": sum_items(row, CG_DISEASE, True),
        "CG_employ_recode": employment_group(row),
        "CG_income_source_count": sum_items(row, CG_INCOME, True),
        "CG_income_CSSA": to_float(row.get("CG_IncomeSource_10", "")),
        "CG_district": category_value(row, "CG_District"),
        "distance_km": to_float(row.get(DISTANCE, "")) / 1000.0,
    }
    cr = to_float(row.get("CRs", ""))
    x["CR_count"] = cr
    x["multiple_CR"] = np.nan if np.isnan(cr) else float(cr >= 2)
    x["CG_other_service_heard"] = to_float(row.get("CG_OtherService", ""))
    x["CG_serviceuse_count"] = sum_items(row, CG_SERVICE_USE, True, "CG_OtherService")
    x["CG_serviceneeds_count"] = sum_items(row, CG_SERVICE_NEEDS, True)

    for group, items in MNST_GROUPS.items():
        x[f"MNST_{group}_score"] = sum_items(row, items, True)
    for column in MNST_ORPHANS:
        x[column] = to_float(row.get(column, ""))

    use_keys = [f"CG_ServiceUse_{i}" for i in range(1, 8)]
    answered = any(not is_blank(row.get(key, "")) for key in use_keys)
    use_count = sum(to_float(row.get(key, "")) == 1 for key in use_keys) if answered else np.nan
    other = to_float(row.get("CG_OtherService", ""))
    if not np.isnan(use_count) and use_count > 0:
        x["service_familiarity"] = "used_services"
    elif other == 1:
        x["service_familiarity"] = "aware_no_recent_use"
    elif other == 0:
        x["service_familiarity"] = "service_newcomer"
    else:
        x["service_familiarity"] = np.nan

    x["use_centre_based"] = sum(to_float(row.get(f"CG_ServiceUse_{i}", "")) == 1 for i in [1, 2]) if answered else np.nan
    x["use_direct_care"] = sum(to_float(row.get(f"CG_ServiceUse_{i}", "")) == 1 for i in [3, 4, 5]) if answered else np.nan
    x["use_hotline"] = float(to_float(row.get("CG_ServiceUse_6", "")) == 1) if answered else np.nan
    x["use_other_type"] = float(to_float(row.get("CG_ServiceUse_7", "")) == 1) if answered else np.nan

    employment = to_float(row.get("CG_Employ", ""))
    if np.isnan(employment):
        context = np.nan
    elif employment in [0, 1, 2]:
        context = "working"
    elif employment == 3:
        context = "homemaker"
    elif employment == 4:
        context = "student"
    elif employment == 5:
        context = "retired"
    elif employment in [6, 7]:
        context = "unemployed_or_seeking"
    elif employment == 8:
        context = "caregiving_prevents_work"
    elif employment == 9:
        context = "health_prevents_work"
    else:
        context = "other"
    x["work_context"] = context
    x["caregiving_blocks_work"] = np.nan if np.isnan(employment) else float(employment == 8)
    x["health_blocks_work"] = np.nan if np.isnan(employment) else float(employment == 9)

    conditions = [
        (to_float(row.get("CG_HouseType", "")) == 5, "CG_HouseType_5_txt"),
        (to_float(row.get("CG_Disease_13", "")) == 1, "CG_Disease_13_txt"),
        (to_float(row.get("CG_Disease_14", "")) == 1, "CG_Disease_14_txt"),
        (to_float(row.get("CG_Employ", "")) == 10, "CG_Employ_10_txt"),
        (to_float(row.get("CG_IncomeSource_11", "")) == 1, "CG_IncomeSource_11_txt"),
        (to_float(row.get("CG_ServiceUse_7", "")) == 1, "CG_ServiceUse_7_txt"),
        (to_float(row.get("CG_ServiceNeeds_8", "")) == 1, "CG_ServiceNeeds_8_txt"),
        (to_float(row.get("CG_InfoSource_16", "")) == 1, "CG_InfoSource_16_txt"),
    ]
    applicable = sum(bool(flag) for flag, _ in conditions)
    completed = sum(1 for flag, text_column in conditions if flag and not is_blank(row.get(text_column, "")))
    x["other_applicable_count"] = applicable
    x["other_text_completed_count"] = completed
    x["other_without_text_count"] = applicable - completed
    x["other_text_completion_rate"] = np.nan if applicable == 0 else completed / applicable
    x[TARGET] = int(to_float(row[TARGET]))
    return x


def load_authorized_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_all = list(csv.DictReader(handle))

    raw = [row for row in raw_all if to_float(row.get(TARGET, "")) in (0, 1) and to_float(row.get(DISTANCE, "")) != 999]
    frame = pd.DataFrame([build_features(row) for row in raw])
    for column in CATEGORICAL:
        frame[column] = frame[column].map(canonical_category)
    for column in set(FEATURES) - CATEGORICAL:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[FEATURES + [TARGET]].copy()
    y = frame[TARGET].astype(int).to_numpy()

    raw_target = pd.Series([to_float(row.get(TARGET, "")) for row in raw_all])
    audit = {
        "raw_rows": len(raw_all),
        "target_999": int((raw_target == 999).sum()),
        "target_0": int((raw_target == 0).sum()),
        "target_1": int((raw_target == 1).sum()),
        "removed_labeled_distance_999": int(sum(to_float(row.get(TARGET, "")) in (0, 1) and to_float(row.get(DISTANCE, "")) == 999 for row in raw_all)),
        "final_rows": len(frame),
        "final_0": int((y == 0).sum()),
        "final_1": int((y == 1).sum()),
    }
    return frame, audit


def balanced_training_indices(y: np.ndarray, train_idx: np.ndarray, fold: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed * 100 + fold)
    class0 = train_idx[y[train_idx] == 0]
    class1 = train_idx[y[train_idx] == 1]
    n = min(len(class0), len(class1))
    out = np.r_[rng.choice(class0, n, replace=False), rng.choice(class1, n, replace=False)]
    rng.shuffle(out)
    return out


def catboost_frame(frame: pd.DataFrame, indices: np.ndarray):
    x = frame.iloc[indices][FEATURES].copy()
    categorical_indices: list[int] = []
    for j, column in enumerate(FEATURES):
        if column in CATEGORICAL:
            categorical_indices.append(j)
            x[column] = x[column].fillna("__MISSING__").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x, categorical_indices


def run_cv(frame: pd.DataFrame, strategy: str, seed: int = 42, n_splits: int = 5, threshold: float = 0.50):
    from catboost import CatBoostClassifier

    y = frame[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics, audits = [], []
    oof = np.full(len(frame), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(frame, y), 1):
        fit_idx = balanced_training_indices(y, train_idx, fold, seed=seed) if strategy == "Balanced" else train_idx
        audits.append({
            "Fold": fold,
            "Train_0_before": int((y[train_idx] == 0).sum()),
            "Train_1_before": int((y[train_idx] == 1).sum()),
            "Train_0_after": int((y[fit_idx] == 0).sum()),
            "Train_1_after": int((y[fit_idx] == 1).sum()),
            "Validation_0": int((y[valid_idx] == 0).sum()),
            "Validation_1": int((y[valid_idx] == 1).sum()),
        })
        x_train, categorical_indices = catboost_frame(frame, fit_idx)
        x_valid, _ = catboost_frame(frame, valid_idx)
        model = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=8,
            random_strength=2,
            bagging_temperature=0.5,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y[fit_idx], cat_features=categorical_indices)
        probabilities = model.predict_proba(x_valid)[:, 1]
        oof[valid_idx] = probabilities
        predicted = (probabilities >= threshold).astype(int)
        fold_metrics.append({
            "Strategy": strategy,
            "Fold": fold,
            "Accuracy": accuracy_score(y[valid_idx], predicted),
            "ROC_AUC": roc_auc_score(y[valid_idx], probabilities),
            "Precision": precision_score(y[valid_idx], predicted, zero_division=0),
            "Recall": recall_score(y[valid_idx], predicted, zero_division=0),
            "F1": f1_score(y[valid_idx], predicted, zero_division=0),
        })

    predicted = (oof >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted)
    return pd.DataFrame(fold_metrics), pd.DataFrame(audits), oof, predicted, matrix


def summarize(strategy: str, folds: pd.DataFrame) -> dict:
    row = {"Strategy": strategy}
    for metric in ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]:
        row[f"{metric}_mean"] = folds[metric].mean()
        row[f"{metric}_std"] = folds[metric].std(ddof=1)
    return row
