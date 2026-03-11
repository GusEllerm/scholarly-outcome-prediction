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
    """Legacy metrics without explicit benchmark_mode/model_family get legacy_inferred classification."""
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
    assert comparison["rows"][0]["benchmark_mode_source"] == "legacy_inferred"
    assert comparison["rows"][0]["model_name"] == "baseline"
    # Legacy artifacts are normalized to canonical model_family vocabulary
    assert comparison["rows"][0]["model_family"] == "trivial_baseline"
    assert comparison["rows"][0]["model_family_source"] == "legacy_inferred"
    assert comparison["rows"][0]["test_zero_rate"] == 0.5
    assert len(comparison["missing"]) > 0


def test_benchmark_comparison_year_conditioned_diagnostic_label() -> None:
    """Legacy metrics: year_conditioned gets legacy_inferred diagnostic baseline and is_diagnostic_only."""
    metrics_list = [
        {
            "experiment_name": "year_conditioned_temporal_h2",
            "model_name": "year_conditioned",
            "dataset_id": "openalex_temporal_articles_1000",
            "target_mode": "calendar_horizon",
            "split_kind": "time",
            "rmse": 1.2,
            "mae": 0.7,
            "r2": -0.4,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["model_family"] == "diagnostic_baseline"
    assert comparison["rows"][0]["model_family_source"] == "legacy_inferred"
    assert comparison["rows"][0]["is_diagnostic_only"] is True
    assert comparison["rows"][0]["is_diagnostic_only_source"] == "legacy_inferred"


def test_ablation_review_no_full_model() -> None:
    """Legacy ablation metrics without explicit ablation_features_removed still readable; source is legacy_inferred."""
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
    assert review["ablations"][0]["features_removed_source"] == "legacy_inferred"
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
    assert ABLATION_FEATURES_REMOVED["no_referenced_works_count"] == ["referenced_works_count"]
    assert ABLATION_FEATURES_REMOVED["no_authors_count"] == ["authors_count"]
    assert ABLATION_FEATURES_REMOVED["no_institutions_count"] == ["institutions_count"]


def test_ablation_review_includes_type_and_tag() -> None:
    """Ablation review rows include ablation_type and interpretation_tag."""
    full = {
        "experiment_name": ABLATION_FULL_EXPERIMENT,
        "model_name": "xgboost",
        "rmse": 0.5,
        "mae": 0.4,
        "r2": 0.3,
    }
    ablated = {
        "experiment_name": "xgb_temporal_h2_no_referenced_works_count",
        "model_name": "xgboost",
        "rmse": 0.6,
        "mae": 0.5,
        "r2": 0.2,
    }
    review = build_ablation_review([full, ablated])
    assert len(review["ablations"]) == 1
    assert review["ablations"][0]["ablation_name"] == "no_referenced_works_count"
    assert review["ablations"][0]["ablation_type"] == "numeric_fine"
    assert review["ablations"][0]["interpretation_tag"] in ("high impact", "moderate impact", "low impact", "negligible / possibly noisy")


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


def test_benchmark_comparison_prefers_explicit_benchmark_mode() -> None:
    """When metrics have explicit benchmark_mode, comparison uses it and records source."""
    metrics_list = [
        {
            "experiment_name": "custom_experiment",
            "model_name": "ridge",
            "benchmark_mode": "representative_h2",
            "dataset_id": "other_dataset",
            "rmse": 0.8,
            "mae": 0.5,
            "r2": 0.2,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["benchmark_mode"] == "representative_h2"
    assert comparison["rows"][0]["benchmark_mode_source"] == "explicit"


def test_benchmark_comparison_prefers_explicit_model_family() -> None:
    """When metrics have explicit model_family, comparison uses it and records source."""
    metrics_list = [
        {
            "experiment_name": "ridge_temporal_h2",
            "model_name": "ridge",
            "benchmark_mode": "temporal_h2",
            "model_family": "linear_baseline",
            "dataset_id": "openalex_temporal_articles_1000",
            "rmse": 0.7,
            "mae": 0.5,
            "r2": 0.4,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["model_family"] == "linear_baseline"
    assert comparison["rows"][0]["model_family_source"] == "explicit"


def test_ablation_review_uses_explicit_features_removed() -> None:
    """When metrics have explicit ablation_features_removed, ablation review uses it (authoritative)."""
    full = {
        "experiment_name": ABLATION_FULL_EXPERIMENT,
        "model_name": "xgboost",
        "rmse": 0.5,
        "mae": 0.4,
        "r2": 0.3,
    }
    ablated = {
        "experiment_name": "xgb_temporal_h2_no_publication_year",
        "model_name": "xgboost",
        "ablation_name": "no_publication_year",
        "ablation_features_removed": ["publication_year"],
        "ablation_type": "coarse",
        "rmse": 0.52,
        "mae": 0.42,
        "r2": 0.28,
    }
    review = build_ablation_review([full, ablated])
    assert len(review["ablations"]) == 1
    assert review["ablations"][0]["features_removed"] == ["publication_year"]
    assert review["ablations"][0]["features_removed_source"] == "explicit"


def test_diagnostic_from_explicit_metadata() -> None:
    """Explicit is_diagnostic_model: true yields is_diagnostic_only and diagnostic family."""
    metrics_list = [
        {
            "experiment_name": "year_conditioned_temporal_h2",
            "model_name": "year_conditioned",
            "benchmark_mode": "temporal_h2",
            "model_family": "diagnostic_baseline",
            "is_diagnostic_model": True,
            "dataset_id": "openalex_temporal_articles_1000",
            "target_mode": "calendar_horizon",
            "rmse": 1.2,
            "mae": 0.7,
            "r2": -0.4,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["is_diagnostic_only"] is True
    assert comparison["rows"][0]["model_family"] == "diagnostic_baseline"
    assert comparison["rows"][0]["is_diagnostic_only_source"] == "explicit"


def test_benchmark_comparison_excludes_tweedie() -> None:
    """Tweedie is excluded from the active benchmark comparison (BENCHMARK_EXCLUDED_MODELS)."""
    metrics_list = [
        {
            "experiment_name": "tweedie_temporal_h2",
            "model_name": "tweedie",
            "benchmark_mode": "temporal_h2",
            "model_family": "count_aware_glm",
            "is_diagnostic_model": False,
            "dataset_id": "openalex_temporal_articles_1000",
            "rmse": 0.7,
            "mae": 0.5,
            "r2": 0.35,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 0
    # Tweedie is not in expected_models, so it may appear in missing; exclusion is that rows are not included
    assert not any(r.get("model_name") == "tweedie" for r in comparison["rows"])


def test_legacy_artifacts_compatibility() -> None:
    """Older metrics JSONs without explicit benchmark_mode/model_family are still readable; classification is legacy_inferred."""
    metrics_list = [
        {
            "experiment_name": "baseline_representative",
            "model_name": "baseline",
            "dataset_id": "openalex_representative_articles_1000",
            "rmse": 1.0,
            "mae": 0.8,
            "r2": 0.0,
        },
    ]
    comparison = build_benchmark_comparison(metrics_list)
    assert len(comparison["rows"]) == 1
    assert comparison["rows"][0]["benchmark_mode"] == "representative_proxy"
    assert comparison["rows"][0]["benchmark_mode_source"] == "legacy_inferred"
    assert comparison["rows"][0]["model_family"] == "trivial_baseline"
    assert comparison["rows"][0]["model_family_source"] == "legacy_inferred"
