from __future__ import annotations

import datetime as dt

import numpy as np

from src.pipeline import balanced_training_indices, parse_age, sum_items, to_float


def test_parse_age_accepts_supported_formats():
    ref = dt.date(2025, 9, 26)
    assert parse_age("26/09/2000", ref) == 25.0
    assert parse_age("2000-09-27", ref) == 24.0
    assert np.isnan(parse_age("not-a-date", ref))


def test_sum_items_respects_missingness_and_fallback():
    row = {"a": "1", "b": "2", "gate": "1"}
    assert sum_items(row, ["a", "b"], True) == 3.0
    assert np.isnan(sum_items({"a": "1", "b": ""}, ["a", "b"], True))
    assert sum_items({"a": "", "b": "", "gate": "0"}, ["a", "b"], True, "gate") == 0.0


def test_balancing_is_training_only_and_one_to_one():
    y = np.array([0] * 8 + [1] * 4)
    train_idx = np.arange(len(y))
    selected = balanced_training_indices(y, train_idx, fold=1)
    assert (y[selected] == 0).sum() == (y[selected] == 1).sum() == 4
    assert set(selected).issubset(set(train_idx))


def test_float_parser_handles_blanks():
    assert to_float("3.5") == 3.5
    assert np.isnan(to_float(""))
    assert np.isnan(to_float("N/A"))


def test_balancing_is_reproducible_for_same_seed():
    y = np.array([0] * 12 + [1] * 6)
    train_idx = np.arange(len(y))
    left = balanced_training_indices(y, train_idx, fold=1, seed=123)
    right = balanced_training_indices(y, train_idx, fold=1, seed=123)
    assert np.array_equal(left, right)
    assert (y[left] == 0).sum() == (y[left] == 1).sum()
