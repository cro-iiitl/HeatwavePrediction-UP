"""
Common model interface — master §10.2.
Every model tier implements this so train.py and src/eval/ never need
tier-specific branches.
"""

from typing import Protocol
import numpy as np


class ForecastModel(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n_samples, 10) — one value per horizon day."""
        ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ForecastModel": ...