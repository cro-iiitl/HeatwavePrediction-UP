"""
Tier 0/1 — naive baselines (master §7), refactored to conform to the
ForecastModel interface (master §10.2) so the walk-forward runner can
treat all tiers uniformly.

IMPORTANT: baselines need actual calendar dates to predict correctly
(climatological/seasonal-naive look up specific past dates), not just
the raw feature window. predict() therefore requires target_dates
alongside X. This is a deliberate interface difference from
LSTM/GBM, which only need X -- documented here rather than silently
forcing baselines into a shape that doesn't fit their logic.

All baselines assume T2M_MAX is column index 0 of the input window's
feature axis (true for both TREE_FEATURE_COLUMNS and
SEQUENCE_FEATURE_COLUMNS, per src/data/windowing.py).
"""

import pickle
import numpy as np
import pandas as pd

T2M_MAX_FEATURE_INDEX = 0  # T2M_MAX is always index 0 in both feature sets
HORIZON_DAYS = 10


class PersistenceModel:
    """Predicts every horizon day as equal to the last observed T2M_MAX
    in the input window. No fitting required."""

    def __init__(self, horizon_days: int = HORIZON_DAYS):
        self.horizon_days = horizon_days

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        pass  # nothing to fit

    def predict(self, X: np.ndarray, target_dates=None) -> np.ndarray:
        last_value = X[:, -1, T2M_MAX_FEATURE_INDEX]  # (n_samples,)
        return np.tile(last_value[:, None], (1, self.horizon_days))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"horizon_days": self.horizon_days}, f)

    @classmethod
    def load(cls, path: str) -> "PersistenceModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(horizon_days=state["horizon_days"])


class RollingMeanModel:
    """Predicts every horizon day as the mean T2M_MAX over the trailing
    `lookback_days` of the input window. No fitting required."""

    def __init__(self, lookback_days: int = 7, horizon_days: int = HORIZON_DAYS):
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        pass

    def predict(self, X: np.ndarray, target_dates=None) -> np.ndarray:
        trailing = X[:, -self.lookback_days:, T2M_MAX_FEATURE_INDEX]  # (n_samples, lookback_days)
        mean_val = trailing.mean(axis=1)
        return np.tile(mean_val[:, None], (1, self.horizon_days))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"lookback_days": self.lookback_days,
                         "horizon_days": self.horizon_days}, f)

    @classmethod
    def load(cls, path: str) -> "RollingMeanModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(**state)


class ClimatologicalNormalModel:
    """Predicts every horizon day using the (month, day) climatological
    mean T2M_MAX, computed from training data ONLY (master §11.5 --
    leakage prevention). This is the one baseline that genuinely fits."""

    def __init__(self, horizon_days: int = HORIZON_DAYS):
        self.horizon_days = horizon_days
        self.normal_table = {}  # (month, day) -> mean T2M_MAX

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            train_dates: list = None) -> None:
        """
        train_dates: list of arrays, one per training sample, giving the
        actual calendar dates of that sample's target span (length
        horizon_days each). Required -- climatological fitting needs real
        dates, not just the feature window.
        """
        if train_dates is None:
            raise ValueError(
                "ClimatologicalNormalModel.fit requires train_dates "
                "(actual calendar dates for y_train) to compute the "
                "(month, day) normal table without leakage."
            )

        records = []
        for date_arr, y_arr in zip(train_dates, y_train):
            for date_str, val in zip(date_arr, y_arr):
                dt = pd.Timestamp(str(date_str))
                records.append({"month": dt.month, "day": dt.day, "value": val})

        df = pd.DataFrame(records)
        table = df.groupby(["month", "day"])["value"].mean()
        self.normal_table = table.to_dict()

    def predict(self, X: np.ndarray, target_dates: list = None) -> np.ndarray:
        if target_dates is None:
            raise ValueError(
                "ClimatologicalNormalModel.predict requires target_dates "
                "(the calendar dates being forecast) to look up the "
                "correct (month, day) normal per horizon day."
            )
        preds = np.zeros((len(target_dates), self.horizon_days))
        for i, date_arr in enumerate(target_dates):
            for h, date_str in enumerate(date_arr):
                dt = pd.Timestamp(str(date_str))
                preds[i, h] = self.normal_table.get((dt.month, dt.day), np.nan)
        return preds

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"horizon_days": self.horizon_days,
                         "normal_table": self.normal_table}, f)

    @classmethod
    def load(cls, path: str) -> "ClimatologicalNormalModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(horizon_days=state["horizon_days"])
        instance.normal_table = state["normal_table"]
        return instance


class SeasonalNaiveModel:
    """Predicts every horizon day using the exact same calendar date one
    year prior, looked up from a historical value table built at fit
    time (training data only)."""

    def __init__(self, horizon_days: int = HORIZON_DAYS):
        self.horizon_days = horizon_days
        self.history_table = {}  # date string -> T2M_MAX value

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            train_dates: list = None) -> None:
        if train_dates is None:
            raise ValueError(
                "SeasonalNaiveModel.fit requires train_dates to build "
                "the historical date -> value lookup table."
            )
        for date_arr, y_arr in zip(train_dates, y_train):
            for date_str, val in zip(date_arr, y_arr):
                self.history_table[date_str] = val

    def predict(self, X: np.ndarray, target_dates: list = None) -> np.ndarray:
        if target_dates is None:
            raise ValueError(
                "SeasonalNaiveModel.predict requires target_dates."
            )
        preds = np.full((len(target_dates), self.horizon_days), np.nan)
        for i, date_arr in enumerate(target_dates):
            for h, date_str in enumerate(date_arr):
                lookup_date = pd.Timestamp(str(date_str)) - pd.DateOffset(years=1)
                lookup_str = lookup_date.strftime("%Y-%m-%d")
                if lookup_str in self.history_table:
                    preds[i, h] = self.history_table[lookup_str]
        return preds

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"horizon_days": self.horizon_days,
                         "history_table": self.history_table}, f)

    @classmethod
    def load(cls, path: str) -> "SeasonalNaiveModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(horizon_days=state["horizon_days"])
        instance.history_table = state["history_table"]
        return instance