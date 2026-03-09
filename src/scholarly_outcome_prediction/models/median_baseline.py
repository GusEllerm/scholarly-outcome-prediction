"""Median baseline regressor: predicts training median (constant)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array


class MedianBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the median of the training target for every sample."""

    def __init__(self) -> None:
        self.median_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MedianBaselineRegressor:
        """Store median of y."""
        y = check_array(y, ensure_2d=False, dtype="numeric")
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return constant prediction."""
        check_array(X)
        return np.full(X.shape[0], self.median_, dtype=float)
