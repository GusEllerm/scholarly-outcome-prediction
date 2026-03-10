"""Model registry: map model name to a callable that returns an estimator."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.base import RegressorMixin

from scholarly_outcome_prediction.models.baseline import BaselineRegressor
from scholarly_outcome_prediction.models.median_baseline import MedianBaselineRegressor
from scholarly_outcome_prediction.models.ridge_model import build_ridge_regressor
from scholarly_outcome_prediction.models.year_conditioned_baseline import YearConditionedBaselineRegressor
from scholarly_outcome_prediction.models.hurdle_baseline import HurdleBaselineRegressor


def _build_xgboost(**kw: Any) -> RegressorMixin:
    from scholarly_outcome_prediction.models.xgboost_model import build_xgboost_regressor

    return build_xgboost_regressor(kw.get("params") or {})


def _build_ridge(**kw: Any) -> RegressorMixin:
    return build_ridge_regressor(kw.get("params") or {})


def _build_year_conditioned(**kw: Any) -> RegressorMixin:
    params = kw.get("params") or {}
    return YearConditionedBaselineRegressor(
        use_mean=params.get("use_mean", False),
        year_column_index=params.get("year_column_index", 0),
    )


def _build_hurdle(**kw: Any) -> RegressorMixin:
    params = kw.get("params") or {}
    return HurdleBaselineRegressor(
        classifier_params=params.get("classifier_params"),
        regressor_params=params.get("regressor_params"),
    )


def get_model_builder(name: str) -> Callable[..., RegressorMixin]:
    """Return a callable that builds a regressor. Raises KeyError if unknown."""
    registry: dict[str, Callable[..., RegressorMixin]] = {
        "baseline": lambda **_: BaselineRegressor(),
        "median_baseline": lambda **_: MedianBaselineRegressor(),
        "ridge": _build_ridge,
        "year_conditioned": _build_year_conditioned,
        "hurdle": _build_hurdle,
        "xgboost": _build_xgboost,
    }
    if name not in registry:
        raise KeyError(f"Unknown model: {name}. Known: {list(registry)}")
    return registry[name]


def list_models() -> list[str]:
    """Return registered model names."""
    return ["baseline", "median_baseline", "ridge", "year_conditioned", "hurdle", "xgboost"]
