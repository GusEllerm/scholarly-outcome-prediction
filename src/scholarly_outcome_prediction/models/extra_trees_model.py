"""ExtraTrees regressor: bagged-tree comparator."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesRegressor


def build_extra_trees_regressor(params: dict[str, Any] | None = None) -> ExtraTreesRegressor:
    """Build an ExtraTreesRegressor; modest defaults for benchmark comparison."""
    defaults: dict[str, Any] = {
        "n_estimators": 50,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "random_state": 42,
    }
    if params:
        defaults.update(params)
    return ExtraTreesRegressor(**defaults)
