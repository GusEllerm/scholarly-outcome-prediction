"""Tests for evaluation metrics: zero-inflation and calibration/tail diagnostics."""

import numpy as np
import pytest

from scholarly_outcome_prediction.evaluation.metrics import (
    compute_calibration_tail_metrics,
    compute_zero_inflation_metrics,
)


def test_zero_inflation_all_zero() -> None:
    y_true = np.zeros(20)
    y_pred = np.random.rand(20)
    out = compute_zero_inflation_metrics(y_true, y_pred)
    assert out["test_zero_rate"] == 1.0
    assert out["test_nonzero_rate"] == 0.0
    assert out["n_zero_target"] == 20
    assert out["n_nonzero_target"] == 0
    assert out["mae_zero_target"] is not None
    assert out["rmse_zero_target"] is not None
    assert out["mae_nonzero_target"] is None
    assert out["rmse_nonzero_target"] is None


def test_zero_inflation_all_nonzero() -> None:
    y_true = np.ones(20) * 2.0
    y_pred = np.ones(20) * 1.5
    out = compute_zero_inflation_metrics(y_true, y_pred)
    assert out["test_zero_rate"] == 0.0
    assert out["test_nonzero_rate"] == 1.0
    assert out["n_zero_target"] == 0
    assert out["n_nonzero_target"] == 20
    assert out["mae_zero_target"] is None
    assert out["rmse_zero_target"] is None
    assert out["mae_nonzero_target"] == 0.5
    assert out["rmse_nonzero_target"] == 0.5


def test_zero_inflation_mixed() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.2, 1.0, 2.0, 3.0])
    out = compute_zero_inflation_metrics(y_true, y_pred)
    assert out["test_zero_rate"] == 0.4
    assert out["test_nonzero_rate"] == 0.6
    assert out["n_zero_target"] == 2
    assert out["n_nonzero_target"] == 3
    assert out["mae_zero_target"] == pytest.approx(0.15)
    assert out["mae_nonzero_target"] == 0.0


def test_calibration_tail_returns_structure() -> None:
    np.random.seed(42)
    y_true = np.abs(np.random.randn(200)) + 0.1
    y_pred = y_true + np.random.randn(200) * 0.3
    out = compute_calibration_tail_metrics(y_true, y_pred, n_deciles=10)
    assert "by_target_decile" in out
    assert "top_quantile_metrics" in out
    assert len(out["by_target_decile"]) <= 10
    for d in out["by_target_decile"]:
        assert "decile" in d and "n" in d and "actual_mean" in d and "pred_mean" in d and "residual_mean" in d and "mae" in d
    for t in out["top_quantile_metrics"]:
        assert "quantile" in t and "n" in t and "mae" in t and "rmse" in t


def test_calibration_tail_top_quantiles() -> None:
    y_true = np.array([1.0] * 80 + [10.0] * 20)
    y_pred = y_true + 0.5
    out = compute_calibration_tail_metrics(y_true, y_pred, top_quantiles=[0.8, 0.9])
    assert len(out["top_quantile_metrics"]) >= 1
    top = out["top_quantile_metrics"]
    for t in top:
        assert t["mae"] >= 0
        assert t["rmse"] >= 0
        assert t["n"] > 0
