"""XGBoost regressor wrapper with configurable params."""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor


def build_xgboost_regressor(params: dict[str, Any] | None = None) -> XGBRegressor:
    """Build an XGBRegressor with given params; defaults suitable for small experiments."""
    defaults: dict[str, Any] = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "auto",
        "random_state": 42,
    }
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)
