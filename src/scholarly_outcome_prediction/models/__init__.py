"""Models and model registry."""

from scholarly_outcome_prediction.models.baseline import BaselineRegressor
from scholarly_outcome_prediction.models.registry import get_model_builder, list_models

__all__ = [
    "BaselineRegressor",
    "get_model_builder",
    "list_models",
]


def __getattr__(name: str):  # lazy import for xgboost (may need libomp on macOS)
    if name == "build_xgboost_regressor":
        from scholarly_outcome_prediction.models.xgboost_model import build_xgboost_regressor

        return build_xgboost_regressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
