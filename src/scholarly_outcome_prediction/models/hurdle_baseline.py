"""Hurdle-style baseline for zero-inflated targets: stage 1 zero vs nonzero, stage 2 regress on positive.

Lightweight two-stage model. Not a full hurdle model; interpretable and minimal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils import check_array


class HurdleBaselineRegressor(BaseEstimator, RegressorMixin):
    """Stage 1: predict zero vs nonzero (binary). Stage 2: Ridge on y > 0. Predict: class zero -> 0 else Ridge."""

    def __init__(
        self,
        classifier_params: dict[str, Any] | None = None,
        regressor_params: dict[str, Any] | None = None,
    ) -> None:
        self.classifier_params = classifier_params or {}
        self.regressor_params = regressor_params or {}
        self.clf_ = LogisticRegression(max_iter=2000, random_state=42, **self.classifier_params)
        self.reg_ = Ridge(alpha=1.0, random_state=42, **self.regressor_params)
        self.positive_mask_ = np.array([], dtype=bool)
        # When train has only one class (all 0 or all 1), we skip the classifier; 0 = always predict 0, 1 = always use regressor.
        self._single_class_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> HurdleBaselineRegressor:
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype="numeric")
        binary = (y > 0).astype(int)
        self.positive_mask_ = y > 0
        n_classes = len(np.unique(binary))
        if n_classes < 2:
            self._single_class_ = int(binary[0])
            if self._single_class_ == 1:
                self.reg_.fit(X, y)
            return self
        self._single_class_ = None
        self.clf_.fit(X, binary)
        if self.positive_mask_.sum() > 0:
            self.reg_.fit(X[self.positive_mask_], y[self.positive_mask_])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_array(X)
        if self._single_class_ == 0:
            return np.zeros(X.shape[0], dtype=float)
        if self._single_class_ == 1:
            out = self.reg_.predict(X).astype(float)
            out[out < 0] = 0
            return out
        pred_zero = self.clf_.predict(X)
        out = np.zeros(X.shape[0], dtype=float)
        nonzero_idx = pred_zero > 0
        if nonzero_idx.sum() > 0 and self.positive_mask_.sum() > 0:
            out[nonzero_idx] = self.reg_.predict(X[nonzero_idx])
            out[out < 0] = 0
        return out
