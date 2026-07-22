"""Evaluation metrics — master §13. Model-agnostic, no imports from src/models/."""

import numpy as np


def rmse_per_horizon_day(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns shape (horizon_days,) -- RMSE computed independently per
    horizon day, not averaged prematurely (master §13.1)."""
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=0))


def skill_score(rmse_model: np.ndarray, rmse_climatology: np.ndarray) -> np.ndarray:
    """Skill Score = 1 - (RMSE_model^2 / RMSE_climatology^2), per horizon
    day (master §8.8, §13.2)."""
    return 1 - (rmse_model ** 2 / rmse_climatology ** 2)