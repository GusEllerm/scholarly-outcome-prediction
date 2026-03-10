"""Tests for benchmark comparison and ablation review generation."""

import tempfile
from pathlib import Path

import pytest

from scholarly_outcome_prediction.evaluation.benchmark_analysis import (
    ABLATION_FEATURES_REMOVED,
    ABLATION_FULL_EXPERIMENT,
    build_ablation_review,
    build_benchmark_comparison,
    load_all_metrics,
    run_benchmark_analysis,
)
from scholarly_outcome_prediction.utils.io import save_json


def test_load_all_metrics_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "metrics").mkdir(parents=True)
        out = load_all_metrics(root)
    assert out == []


def test_load_all_metrics_with_files() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True)
        save_json(
            {"experiment_name": "xgb_temporal_h2", "model_name": "xgboost", "rmse": 0.5, "target_mode": "calendar_horizon"},
            metrics_dir / "xgb_temporal_h2.json",
        )
        out = load_all_metrics(root)
    assert len(out) == 1
    assert out[0]["experiment_name"] == "xgb_temporal_h2"
    assert out[0]["rmse"] == 0.5


def test_benchmark_comparison_with_partial_missing() -> None:
    """With no metrics, comparison has empty rows and full missing list."""
    comparison = build_benchmark_comparison([])
    assert comparison["report_scope"] == "benchmark_comparison"
    assert "generated_at" in comparison
    assert comparison["rows"] == []
    assert len(comparison["missing"]) > 0
    # Missing should include at least one entry per benchmark mode × model
    missing_modes = {m["benchmark_mode"] for m in comparison["missing"]}
    assert "temporal_h2" in missing_modes or "representative_proxy" in missing_modes


def test_benchmark_comparison_one_run() -> None:
    metrics_list = [
        {
            "experiment_name": "baseline_temporal_h2",
            "model_name": "baseline",
            "dataset_id": "openalex_temporal_articles_1000",
            "target_mode": "calendar_horizon",
            "split_kind": "time",
            "rmse": 1.0,
            "mae": 0.8,
            "r2": 0.1,
            "zero_inflation": {"test_zero_rate": 0.5, "mae_zero_target": 0.3, "mae_nonzero_target": 1.0},
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["benchmark_mode"] == "temporal_h2"
    assert comparison["rows"][0]["model_name"] == "baseline"
    assert comparison["rows"][0]["test_zero_rate"] == 0.5
    assert len(comparison["missing"]) > 0


def test_ablation_review_no_full_model() -> None:
    """Ablation review with only ablation runs (no full xgb_temporal_h2) still runs."""
    metrics_list = [
        {
            "experiment_name": "xgb_temporal_h2_no_publication_year",
            "model_name": "xgboost",
            "rmse": 0.6,
            "mae": 0.5,
            "r2": 0.2,
        },
    ]
    review = build_ablation_review(metrics_list)
    assert review["report_scope"] == "ablation_review"
    assert review["full_model_available"] is False
    assert len(review["ablations"]) == 1
    assert review["ablations"][0]["ablation_name"] == "no_publication_year"
    assert review["ablations"][0]["features_removed"] == ABLATION_FEATURES_REMOVED["no_publication_year"]
    assert review["ablations"][0]["delta_rmse"] is None


def test_ablation_review_with_full_model() -> None:
    full = {
        "experiment_name": ABLATION_FULL_EXPERIMENT,
        "model_name": "xgboost",
        "rmse": 0.5,
        "mae": 0.4,
        "r2": 0.3,
    }
    ablated = {
        "experiment_name": "xgb_temporal_h2_no_venue_name",
        "model_name": "xgboost",
        "rmse": 0.55,
        "mae": 0.45,
        "r2": 0.25,
    }
    review = build_ablation_review([full, ablated])
    assert review["full_model_available"] is True
    assert len(review["ablations"]) == 1
    assert review["ablations"][0]["delta_rmse"] == pytest.approx(0.05)
    assert review["ablations"][0]["delta_r2"] == pytest.approx(-0.05)


def test_ablation_features_removed_mapping() -> None:
    """Ablation config mapping covers expected ablations."""
    assert "no_publication_year" in ABLATION_FEATURES_REMOVED
    assert ABLATION_FEATURES_REMOVED["no_publication_year"] == ["publication_year"]
    assert "numeric_only" in ABLATION_FEATURES_REMOVED
    assert "categorical_only" in ABLATION_FEATURES_REMOVED


def test_run_benchmark_analysis_writes_artifacts() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "metrics").mkdir(parents=True)
        save_json(
            {"experiment_name": "xgb_temporal_h2", "model_name": "xgboost", "rmse": 0.5, "target_mode": "calendar_horizon"},
            root / "metrics" / "xgb_temporal_h2.json",
        )
        out_dir = root / "reports"
        summary = run_benchmark_analysis(root, out_dir=out_dir)
        assert summary["metrics_loaded"] == 1
        assert summary["comparison_rows"] == 1
        assert (out_dir / "benchmark_comparison.json").exists()
        assert (out_dir / "benchmark_comparison.md").exists()
        assert (out_dir / "ablation_review.json").exists()
        assert (out_dir / "ablation_review.md").exists()
