"""
Common model interface — master §10.2.
Every model tier implements this so train.py and src/eval/ never need
tier-specific branches.

train_dates/target_dates are optional, date-carrying parameters needed
only by baselines that look up calendar-specific history
(ClimatologicalNormalModel, SeasonalNaiveModel). LSTM/GBM ignore them
entirely. Widening the Protocol this way (rather than special-casing
baselines in the runner) keeps exactly one interface every tier
implements, consistent with this section's own stated purpose.
"""

from typing import Protocol, Optional, List
import numpy as np


class ForecastModel(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            train_dates: Optional[List[np.ndarray]] = None) -> None:
        """
        train_dates: only required by date-lookup baselines (climatological
        normal, seasonal-naive) -- one array of calendar date strings per
        training sample, matching y_train's target span. Ignored by
        tiers that don't need it (LSTM, GBM, persistence, rolling-mean).
        """
        ...

    def predict(self, X: np.ndarray,
                target_dates: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Returns shape (n_samples, 10) -- one value per horizon day.
        target_dates: only required by date-lookup baselines. Ignored
        by tiers that don't need it.
        """
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ForecastModel": ...