"""Year-conditioned baseline: predicts year-specific median (or mean) of the training target.

Requires the first numeric feature to be publication_year (or a year column) in the same
order as in the experiment config. Used to test whether models are mostly exploiting time/cohort.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array


class YearConditionedBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the median of the training target per publication year.

    Expects X's first column to be publication_year (or the same year used for grouping).
    Rows with year not seen in training get the global training median.
    """

    def __init__(self, use_mean: bool = False, year_column_index: int = 0) -> None:
        self.use_mean = use_mean
        self.year_column_index = year_column_index
        self.by_year_: dict[float, float] = {}
        self.global_fallback_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> YearConditionedBaselineRegressor:
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype="numeric")
        if X.shape[0] != len(y):
            raise ValueError("X and y length mismatch")
        year_col = X[:, self.year_column_index]
        self.global_fallback_ = float(np.median(y)) if not self.use_mean else float(np.mean(y))
        self.by_year_ = {}
        for yr in np.unique(year_col):
            mask = year_col == yr
            subset = y[mask]
            if len(subset):
                self.by_year_[float(yr)] = float(np.median(subset)) if not self.use_mean else float(np.mean(subset))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_array(X)
        year_col = X[:, self.year_column_index]
        out = np.full(X.shape[0], self.global_fallback_, dtype=float)
        for yr, val in self.by_year_.items():
            out[year_col == yr] = val
        return out
