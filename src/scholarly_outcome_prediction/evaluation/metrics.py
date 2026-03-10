"""Regression metrics: RMSE, MAE, R²; zero-inflation and calibration/tail diagnostics."""

from __future__ import annotations

from typing import Any

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


def compute_zero_inflation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Standard zero-inflation slice: rates and MAE/RMSE on zero-target vs nonzero-target subsets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    zero_mask = y_true == 0
    nonzero_mask = ~zero_mask
    n_zero = int(zero_mask.sum())
    n_nonzero = int(nonzero_mask.sum())
    n = len(y_true)
    out: dict[str, Any] = {
        "test_zero_rate": round(float(n_zero / n), 4) if n else 0.0,
        "test_nonzero_rate": round(float(n_nonzero / n), 4) if n else 0.0,
        "n_zero_target": n_zero,
        "n_nonzero_target": n_nonzero,
    }
    if n_zero > 0:
        out["mae_zero_target"] = float(mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]))
        out["rmse_zero_target"] = float(np.sqrt(mean_squared_error(y_true[zero_mask], y_pred[zero_mask])))
    else:
        out["mae_zero_target"] = None
        out["rmse_zero_target"] = None
    if n_nonzero > 0:
        out["mae_nonzero_target"] = float(mean_absolute_error(y_true[nonzero_mask], y_pred[nonzero_mask]))
        out["rmse_nonzero_target"] = float(np.sqrt(mean_squared_error(y_true[nonzero_mask], y_pred[nonzero_mask])))
    else:
        out["mae_nonzero_target"] = None
        out["rmse_nonzero_target"] = None
    return out


def compute_calibration_tail_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 10,
    top_quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """Bucketed diagnostics: by target decile, and MAE on top quantiles of target (tail)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if top_quantiles is None:
        top_quantiles = [0.9, 0.95, 0.99]
    out: dict[str, Any] = {}
    # By decile of actual target
    if len(y_true) >= n_deciles:
        decile_edges = np.percentile(y_true, np.linspace(0, 100, n_deciles + 1))
        by_decile = []
        for i in range(n_deciles):
            low, high = decile_edges[i], decile_edges[i + 1]
            mask = (y_true >= low) & (y_true <= high) if i == n_deciles - 1 else (y_true >= low) & (y_true < high)
            if mask.sum() == 0:
                continue
            by_decile.append({
                "decile": i + 1,
                "n": int(mask.sum()),
                "actual_mean": float(np.mean(y_true[mask])),
                "pred_mean": float(np.mean(y_pred[mask])),
                "residual_mean": float(np.mean(y_pred[mask] - y_true[mask])),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            })
        out["by_target_decile"] = by_decile
    # Top quantiles of target (tail performance)
    tail_metrics = []
    for q in top_quantiles:
        thresh = np.nanquantile(y_true, q)
        mask = y_true >= thresh
        if mask.sum() > 0:
            tail_metrics.append({
                "quantile": q,
                "n": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            })
    out["top_quantile_metrics"] = tail_metrics
    return out
