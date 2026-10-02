from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    cross_fitted_target_encode,
    engineer,
    validate_bundle,
    validate_submission,
)


def small_frames():
    train = pd.DataFrame(
        {
            "id": range(1, 11),
            "Age": [20, 21, 22, 23, 24, 35, 36, 37, 38, 39],
            "Annual_Income_USD": [30000, 32000, 34000, 36000, 38000, 60000, 62000, 64000, 66000, 68000],
            "Daily_Commute_km": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "City_Type": ["Urban"] * 5 + ["Rural"] * 5,
            "Will_Buy_EV": ["No", "Yes"] * 5,
        }
    )
    test = train.drop(columns="Will_Buy_EV").iloc[:4].copy()
    test["id"] = [101, 102, 103, 104]
    sample = pd.DataFrame({"id": test["id"], "Will_Buy_EV": 0.5})
    return train, test, sample


def test_schema_and_feature_engineering():
    train, test, sample = small_frames()
    bundle = validate_bundle(train, test, sample)
    assert bundle.target.tolist() == [0, 1] * 5
    engineered = engineer(train, bundle.features)
    assert "income_log" in engineered
    assert "income_per_age" in engineered
    assert "commute_bin_5" in engineered


def test_nested_target_encoding_handles_unseen_category():
    train, _, _ = small_frames()
    x = train[["City_Type"]].copy()
    y = np.array([0, 1] * 5, dtype="int8")
    valid = pd.DataFrame({"City_Type": ["Urban", "NeverSeen"]})
    test = pd.DataFrame({"City_Type": ["Rural", "NeverSeen"]})
    a, b, t = cross_fitted_target_encode(x, valid, test, y, ["City_Type"], inner_folds=2)
    assert np.isfinite(a["City_Type_te"]).all()
    assert np.isfinite(b["City_Type_te"]).all()
    assert np.isfinite(t["City_Type_te"]).all()
    assert t.loc[1, "City_Type_te"] == pytest.approx(y.mean())


def test_submission_validation_rejects_bad_range():
    train, test, sample = small_frames()
    good = sample.copy()
    good["Will_Buy_EV"] = [0.1, 0.3, 0.7, 0.9]
    validate_submission(good, sample, test)

    bad = good.copy()
    bad.loc[0, "Will_Buy_EV"] = 1.5
    with pytest.raises(ValueError, match="range"):
        validate_submission(bad, sample, test)
