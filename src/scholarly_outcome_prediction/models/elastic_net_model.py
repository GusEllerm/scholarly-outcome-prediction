"""ElasticNet regression: sparse vs dense linear comparator."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import ElasticNet


def build_elastic_net_regressor(params: dict[str, Any] | None = None) -> ElasticNet:
    """Build an ElasticNet regressor; sensible defaults for benchmark comparison with ridge."""
    defaults: dict[str, Any] = {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "random_state": 42,
        "max_iter": 2000,
    }
    if params:
        defaults.update(params)
    return ElasticNet(**defaults)
