"""Ridge regression model with configurable params."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import Ridge


def build_ridge_regressor(params: dict[str, Any] | None = None) -> Ridge:
    """Build a Ridge regressor with given params; sensible defaults for small experiments."""
    defaults: dict[str, Any] = {
        "alpha": 1.0,
        "random_state": 42,
        "solver": "auto",
    }
    if params:
        defaults.update(params)
    return Ridge(**defaults)
