"""Baseline regressor: predicts training mean (constant)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array


class BaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the mean of the training target for every sample."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineRegressor:
        """Store mean of y."""
        y = check_array(y, ensure_2d=False, dtype="numeric")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return constant prediction."""
        check_array(X)
        return np.full(X.shape[0], self.mean_, dtype=float)
