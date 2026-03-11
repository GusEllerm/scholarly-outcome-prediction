"""Tweedie regressor: count-aware GLM for non-negative citation-like targets."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import TweedieRegressor


def build_tweedie_regressor(params: dict[str, Any] | None = None) -> TweedieRegressor:
    """Build a TweedieRegressor for non-negative targets (e.g. citation counts).

    Default power=1.5 (compound Poisson) is a reasonable choice for over-dispersed counts.
    Use power=1 for Poisson. Target should be non-negative; log1p transform in config is typical.
    """
    defaults: dict[str, Any] = {
        "power": 1.5,
        "alpha": 1.0,
        "max_iter": 2000,
        "link": "log",
    }
    if params:
        defaults.update(params)
    return TweedieRegressor(**defaults)
