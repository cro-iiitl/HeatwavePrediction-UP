"""
Walk-forward fold definitions and per-fold data preparation — master
§11.3. Fold boundaries are config, not code, per master §9's own
philosophy -- defined here as data, imported by the runner.
"""

import pandas as pd
import numpy as np

FOLDS = [
    {"fold_id": 1, "train_start": "1984-01-01", "train_end": "2010-12-31",
     "test_start": "2011-01-01", "test_end": "2015-12-31"},
    {"fold_id": 2, "train_start": "1984-01-01", "train_end": "2015-12-31",
     "test_start": "2016-01-01", "test_end": "2020-12-31"},
    {"fold_id": 3, "train_start": "1984-01-01", "train_end": "2020-12-31",
     "test_start": "2021-01-01", "test_end": "2025-12-31"},
]


def expand_target_dates(target_start_dates: list, horizon_days: int) -> list:
    """
    Given a list of target-window start dates (one per sample), return a
    list of full date arrays (each length horizon_days), needed by
    ClimatologicalNormalModel / SeasonalNaiveModel, which require every
    individual forecast date, not just the window's start.
    """
    expanded = []
    for start_str in target_start_dates:
        start = pd.Timestamp(start_str)
        dates = pd.date_range(start, periods=horizon_days, freq="D")
        expanded.append(np.array([d.strftime("%Y-%m-%d") for d in dates]))
    return expanded


def split_windows_by_fold(window_sets: list, fold: dict):
    """
    window_sets: list of WindowSet (one per district), from
    src/data/windowing.build_windows_multi_district.
    fold: one entry from FOLDS.

    Returns (X_train, y_train, train_target_dates,
             X_test, y_test, test_target_dates)
    A window belongs to train if its target span falls ENTIRELY within
    [train_start, train_end]; to test if entirely within
    [test_start, test_end]. Windows whose target spans a boundary are
    excluded from both (master §11.5).
    """
    X_train_list, y_train_list, train_dates_list = [], [], []
    X_test_list, y_test_list, test_dates_list = [], [], []

    for ws in window_sets:
        for j in range(len(ws.input_start_dates)):
            t_start = ws.target_start_dates[j]
            t_end = ws.target_end_dates[j]

            if fold["train_start"] <= t_start and t_end <= fold["train_end"]:
                X_train_list.append(ws.X[j])
                y_train_list.append(ws.y[j])
                train_dates_list.append(t_start)
            elif fold["test_start"] <= t_start and t_end <= fold["test_end"]:
                X_test_list.append(ws.X[j])
                y_test_list.append(ws.y[j])
                test_dates_list.append(t_start)

    X_train = np.stack(X_train_list)
    y_train = np.stack(y_train_list)
    X_test = np.stack(X_test_list)
    y_test = np.stack(y_test_list)

    horizon_days = y_train.shape[1]
    train_target_dates = expand_target_dates(train_dates_list, horizon_days)
    test_target_dates = expand_target_dates(test_dates_list, horizon_days)

    return X_train, y_train, train_target_dates, X_test, y_test, test_target_dates