"""Regression metrics: RMSE, MAE, R²."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    return float(r2_score(y_true, y_pred))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute requested regression metrics; default: rmse, mae, r2."""
    if metric_names is None:
        metric_names = ["rmse", "mae", "r2"]
    out: dict[str, float] = {}
    for name in metric_names:
        n = name.lower()
        if n == "rmse":
            out["rmse"] = compute_rmse(y_true, y_pred)
        elif n == "mae":
            out["mae"] = compute_mae(y_true, y_pred)
        elif n == "r2":
            out["r2"] = compute_r2(y_true, y_pred)
    return out
