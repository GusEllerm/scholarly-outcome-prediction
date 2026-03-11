"""HistGradientBoosting regressor: scikit-learn-native boosting comparator."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor


def build_hist_gradient_boosting_regressor(
    params: dict[str, Any] | None = None,
) -> HistGradientBoostingRegressor:
    """Build a HistGradientBoostingRegressor; conservative defaults for benchmark comparison."""
    defaults: dict[str, Any] = {
        "max_iter": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 42,
    }
    if params:
        defaults.update(params)
    return HistGradientBoostingRegressor(**defaults)
