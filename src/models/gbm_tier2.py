"""
Tier 2 — GBM, flattened window (master §7).
LightGBM, one independent model per horizon day (direct multi-step,
non-shared representation -- distinct from Tier 3's shared-hidden-state
approach). Uses TREE_FEATURE_COLUMNS (includes T2M_MAX's own lags/rolling
stats), since GBM has no internal memory of its own and needs those
features explicitly, unlike the LSTM's SEQUENCE_FEATURE_COLUMNS.
"""

import pickle
import numpy as np
import lightgbm as lgb


class GBMForecastModel:
    """Implements the ForecastModel interface (src/models/interfaces.py)."""

    def __init__(self, horizon_days: int = 10, n_estimators: int = 200,
                 max_depth: int = -1, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 20,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 random_state: int = 42, n_jobs: int = -1, verbose: int = -1):
        self.horizon_days = horizon_days
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=random_state,
            n_jobs=n_jobs, verbose=verbose,
        )
        self.models = []  # one LGBMRegressor per horizon day

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """(n_samples, window_length, n_features) -> (n_samples, window_length * n_features)"""
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            train_dates=None) -> None:
        # train_dates accepted for interface consistency, unused by GBM
        X_flat = self._flatten(X_train)
        self.models = []

        for h in range(self.horizon_days):
            model = lgb.LGBMRegressor(**self.params)

            eval_set = None
            callbacks = []
            if X_val is not None and y_val is not None:
                X_val_flat = self._flatten(X_val)
                eval_set = [(X_val_flat, y_val[:, h])]
                callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

            model.fit(
                X_flat, y_train[:, h],
                eval_set=eval_set,
                callbacks=callbacks if eval_set else None,
            )
            self.models.append(model)

    def predict(self, X, target_dates=None) -> np.ndarray:
        # target_dates accepted for interface consistency, unused by GBM
        X_flat = self._flatten(X)
        preds = np.stack([m.predict(X_flat) for m in self.models], axis=1)
        return preds

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "params": self.params,
                         "horizon_days": self.horizon_days}, f)

    @classmethod
    def load(cls, path: str) -> "GBMForecastModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(horizon_days=state["horizon_days"], **state["params"])
        instance.models = state["models"]
        return instance