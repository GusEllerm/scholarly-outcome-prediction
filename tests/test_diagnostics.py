"""Tests for diagnostic/report correctness: scopes, pipeline trace, artifact audit, shared stats, overlap audit."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scholarly_outcome_prediction.diagnostics import (
    build_pipeline_trace,
    build_pipeline_trace_from_run_context,
    audit_run_artifacts,
    profile_dataset,
    compute_overlap_report,
    run_overlap_audit,
)
from scholarly_outcome_prediction.diagnostics.dataset_stats import compute_canonical_dataset_stats
from scholarly_outcome_prediction.validation.dataset_validation import validate_processed_dataset


def test_pipeline_trace_design_has_scope_and_name() -> None:
    """Design-scoped pipeline trace includes report_scope and report_name."""
    trace = build_pipeline_trace()
    assert trace.get("report_scope") == "design"
    assert trace.get("report_name") == "pipeline_trace"
    assert "generated_at" in trace
    assert "steps" in trace


def test_pipeline_trace_from_run_context_includes_fetch_controls() -> None:
    """Run-scoped pipeline trace includes effective fetch controls (work_types, stratify, sampling)."""
    class DataCfg:
        dataset_name = "test_dataset"
        output_path = "data/raw/test.jsonl"
        sample_size = 500
        from_publication_date = "2018-01-01"
        to_publication_date = "2020-12-31"
        seed = 42
        work_types = ["article"]
        stratify_by_year = True
        use_random_sample = True

    class ExpCfg:
        experiment_name = "baseline"
        model = type("M", (), {"name": "baseline"})()
        split = type("S", (), {"split_kind": "random"})()
        target = type("T", (), {"name": "cited_by_count", "target_mode": "proxy", "transform": "log1p"})()
        features = type("F", (), {"numeric": [], "categorical": []})()
        data = type("D", (), {"processed_path": "data/processed/test.parquet", "model_dump": lambda self: {"processed_path": "data/processed/test.parquet"}})()

    trace = build_pipeline_trace_from_run_context(
        run_id="2025-01-15T12:00:00Z",
        data_config_path=Path("configs/data/test.yaml"),
        data_cfg=DataCfg(),
        baseline_config_path=Path("configs/experiments/baseline.yaml"),
        base_cfg=ExpCfg(),
        xgb_config_path=Path("configs/experiments/xgb.yaml"),
        xgb_cfg=ExpCfg(),
        effective_processed_path=Path("data/processed/test_dataset.parquet"),
        validation_json_path=Path("artifacts/reports/test_dataset_dataset_validation.json"),
        stages_completed={"fetch": True, "prepare": True, "validation": True, "train": True, "evaluate": True},
        metrics_paths=[],
        model_paths=[],
        dataset_id="test_dataset",
    )
    assert trace.get("report_scope") == "run"
    assert trace.get("run_id") == "2025-01-15T12:00:00Z"
    assert trace.get("dataset_id") == "test_dataset"
    data = trace.get("data_config", {})
    assert data.get("work_types") == ["article"]
    assert data.get("stratify_by_year") is True
    assert data.get("use_random_sample") is True
    assert "effective_sampling_strategy" in data
    assert "representative_vs_temporal" in data


def test_pipeline_trace_cross_check_dataset_id_from_metrics(tmp_path: Path) -> None:
    """Run-scoped pipeline trace loads metrics JSONs and fills data_config_dataset_id_equals_metrics_dataset_id."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    # Matching dataset_id
    (metrics_dir / "baseline.json").write_text(
        json.dumps({"effective_dataset_id": "my_dataset", "rmse": 1.0}), encoding="utf-8"
    )
    # Mismatch
    (metrics_dir / "xgb.json").write_text(
        json.dumps({"effective_dataset_id": "other_dataset", "rmse": 0.9}), encoding="utf-8"
    )

    class DataCfg:
        dataset_name = "my_dataset"
        output_path = "data/raw/test.jsonl"
        sample_size = 100
        from_publication_date = "2019-01-01"
        to_publication_date = "2019-12-31"
        seed = 42

    class ExpCfg:
        experiment_name = "baseline"
        model = type("M", (), {"name": "baseline"})()
        split = type("S", (), {"split_kind": "random"})()
        target = type("T", (), {"name": "cited_by_count", "target_mode": "proxy", "transform": "log1p"})()
        features = type("F", (), {"numeric": [], "categorical": []})()
        data = type("D", (), {"processed_path": "data/processed/test.parquet", "model_dump": lambda self: {"processed_path": "data/processed/test.parquet"}})()

    trace = build_pipeline_trace_from_run_context(
        run_id="2025-01-15T12:00:00Z",
        data_config_path=Path("configs/data/test.yaml"),
        data_cfg=DataCfg(),
        baseline_config_path=Path("configs/experiments/baseline.yaml"),
        base_cfg=ExpCfg(),
        xgb_config_path=Path("configs/experiments/xgb.yaml"),
        xgb_cfg=ExpCfg(),
        effective_processed_path=Path("data/processed/my_dataset.parquet"),
        validation_json_path=None,
        stages_completed={"fetch": True, "prepare": True, "validation": True, "train": True, "evaluate": True},
        metrics_paths=[metrics_dir / "baseline.json", metrics_dir / "xgb.json"],
        model_paths=[],
        dataset_id="my_dataset",
    )
    cross = trace.get("cross_checks", {}).get("data_config_dataset_id_equals_metrics_dataset_id")
    assert cross is not None
    assert cross.get("data_config_dataset_id") == "my_dataset"
    assert cross.get("all_agree") is False
    per = cross.get("per_metrics_file", {})
    assert per.get("baseline.json", {}).get("metrics_effective_dataset_id") == "my_dataset"
    assert per.get("baseline.json", {}).get("agree") is True
    assert per.get("xgb.json", {}).get("metrics_effective_dataset_id") == "other_dataset"
    assert per.get("xgb.json", {}).get("agree") is False


def test_artifact_audit_run_scope_has_run_id() -> None:
    """When run_id is passed, artifact audit is run-scoped and includes run_id."""
    audit = audit_run_artifacts(Path("/nonexistent"), run_id="my_run", dataset_id="my_dataset")
    assert audit.get("report_scope") == "run"
    assert audit.get("report_name") == "run_artifact_audit"
    assert audit.get("run_id") == "my_run"
    assert "generated_at" in audit
    assert "metrics_expected_not_found" in audit or "metrics_found" in audit


def test_dataset_profile_has_scope_and_uses_canonical_stats(tmp_path: Path) -> None:
    """Dataset profile is dataset-scoped and includes canonical stats keys."""
    path = tmp_path / "test.parquet"
    df = pd.DataFrame({
        "publication_year": [2019] * 10,
        "type": ["article"] * 10,
        "cited_by_count": [1.0] * 10,
        "venue_name": ["J"] * 10,
    })
    df.to_parquet(path, index=False)
    profile = profile_dataset(path, dataset_id="test_id")
    assert profile.get("report_scope") == "dataset"
    assert profile.get("report_name") == "dataset_profile"
    assert profile.get("dataset_id") == "test_id"
    assert "row_count" in profile
    assert "publication_year" in profile
    assert "citation_distribution" in profile
    assert "missingness_summary" in profile
    assert "categorical_tops" in profile


def test_design_pipeline_trace_no_placeholder_config_paths() -> None:
    """Design-scoped pipeline trace has no placeholder config path strings."""
    trace = build_pipeline_trace()
    assert trace.get("config_paths") is None
    assert "config_paths_note" in trace or "design_note" in trace
    assert "data config" not in str(trace.get("config_paths", ""))
    assert "baseline experiment config" not in str(trace.get("config_paths", ""))


def test_run_scoped_trace_has_consistency_checks() -> None:
    """Run-scoped pipeline trace includes machine-readable consistency_checks (pass/fail/unknown)."""
    class DataCfg:
        dataset_name = "d"
        output_path = "data/raw/d.jsonl"
        sample_size = 10
        from_publication_date = "2019-01-01"
        to_publication_date = "2019-12-31"
        seed = 42

    class ExpCfg:
        experiment_name = "b"
        model = type("M", (), {"name": "b"})()
        split = type("S", (), {"split_kind": "random"})()
        target = type("T", (), {"name": "y", "target_mode": "proxy", "transform": "log1p"})()
        features = type("F", (), {"numeric": [], "categorical": []})()
        data = type("D", (), {"processed_path": "data/processed/d.parquet", "model_dump": lambda self: {"processed_path": "data/processed/d.parquet"}})()

    trace = build_pipeline_trace_from_run_context(
        run_id="run-1",
        data_config_path=Path("configs/data/d.yaml"),
        data_cfg=DataCfg(),
        baseline_config_path=Path("configs/experiments/b.yaml"),
        base_cfg=ExpCfg(),
        xgb_config_path=Path("configs/experiments/x.yaml"),
        xgb_cfg=ExpCfg(),
        effective_processed_path=Path("data/processed/d.parquet"),
        validation_json_path=None,
        stages_completed={"fetch": True, "prepare": True, "validation": True, "train": True, "evaluate": True},
        metrics_paths=[],
        model_paths=[],
        dataset_id="d",
    )
    checks = trace.get("consistency_checks")
    assert checks is not None
    assert "dataset_id_match" in checks
    assert "processed_path_match" in checks
    assert "validation_input_match" in checks
    assert "baseline_xgb_metadata_match" in checks
    assert "artifacts_present" in checks
    for v in checks.values():
        assert v in ("pass", "fail", "unknown")


def test_run_scoped_trace_config_paths_are_resolved() -> None:
    """Run-scoped pipeline trace config_paths are actual paths, not placeholders."""
    class DataCfg:
        dataset_name = "d"
        output_path = "data/raw/d.jsonl"
        sample_size = 10
        from_publication_date = "2019-01-01"
        to_publication_date = "2019-12-31"
        seed = 42

    class ExpCfg:
        experiment_name = "b"
        model = type("M", (), {"name": "b"})()
        split = type("S", (), {"split_kind": "random"})()
        target = type("T", (), {"name": "y", "target_mode": "proxy", "transform": "log1p"})()
        features = type("F", (), {"numeric": [], "categorical": []})()
        data = type("D", (), {"processed_path": "data/processed/d.parquet", "model_dump": lambda self: {"processed_path": "data/processed/d.parquet"}})()

    trace = build_pipeline_trace_from_run_context(
        run_id="run-1",
        data_config_path=Path("configs/data/d.yaml"),
        data_cfg=DataCfg(),
        baseline_config_path=Path("configs/experiments/b.yaml"),
        base_cfg=ExpCfg(),
        xgb_config_path=Path("configs/experiments/x.yaml"),
        xgb_cfg=ExpCfg(),
        effective_processed_path=Path("data/processed/d.parquet"),
        validation_json_path=None,
        stages_completed={"fetch": True, "prepare": True, "validation": True, "train": True, "evaluate": True},
        metrics_paths=[],
        model_paths=[],
        dataset_id="d",
    )
    config_paths = trace.get("config_paths")
    assert config_paths is not None
    assert "data" in config_paths
    assert "baseline_experiment" in config_paths
    assert "xgb_experiment" in config_paths
    assert config_paths["data"] is None or ("configs" in config_paths["data"] or str(Path(config_paths["data"]).resolve()) == config_paths["data"])
    assert "data config" not in str(config_paths)
    assert "baseline config" not in str(config_paths).lower() or "baseline_experiment" in config_paths


def test_preprocessing_audit_static_findings_not_verified_run() -> None:
    """Design-scoped preprocessing audit uses static_audit_findings, not verified_run_facts."""
    from scholarly_outcome_prediction.diagnostics.preprocessing_audit import audit_preprocessing_and_leakage

    audit = audit_preprocessing_and_leakage()
    assert audit.get("report_scope") == "design"
    assert "verified_run_facts" not in audit
    assert "static_audit_findings" in audit
    assert "inferred_from" in audit.get("static_audit_findings", {})


def test_artifact_audit_dataset_scope_has_audit_id_no_run_id() -> None:
    """When run_id is not passed, artifact audit is dataset-scoped with audit_id and no run_id."""
    audit = audit_run_artifacts(Path("/nonexistent"), run_id=None, dataset_id="my_dataset")
    assert audit.get("report_scope") == "dataset"
    assert "run_id" not in audit
    assert audit.get("audit_id") is not None
    assert "generated_at" in audit


def test_validation_and_profile_share_canonical_stats_keys() -> None:
    """Validation and profile both get row_count, publication_year, citation_distribution, missingness from same source."""
    df = pd.DataFrame({
        "publication_year": [2018] * 50 + [2019] * 50,
        "type": ["article"] * 100,
        "cited_by_count": [10.0] * 100,
        "venue_name": ["J"] * 100,
    })
    stats = compute_canonical_dataset_stats(df)
    validation = validate_processed_dataset(df, min_row_count=10, min_years_with_data=2)
    assert stats["row_count"] == validation["row_count"]
    assert stats.get("publication_year") == validation.get("publication_year")
    assert stats.get("citation_distribution") == validation.get("citation_distribution")
    assert stats.get("missingness_summary") == validation.get("missingness_summary")


def test_overlap_audit_identical_datasets() -> None:
    """Overlap audit reports identical=True when both datasets have the same work IDs."""
    df = pd.DataFrame({"openalex_id": ["W1", "W2", "W3"], "x": [1, 2, 3]})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        left_path = path / "left.parquet"
        right_path = path / "right.parquet"
        df.to_parquet(left_path, index=False)
        df.to_parquet(right_path, index=False)
        report = compute_overlap_report(left_path, right_path)
    assert report.get("error") is None
    assert report["size_left"] == 3
    assert report["size_right"] == 3
    assert report["overlap_count"] == 3
    assert report["identical"] is True
    assert report["only_in_left_count"] == 0
    assert report["only_in_right_count"] == 0


def test_overlap_audit_partial_overlap() -> None:
    """Overlap audit reports correct counts when datasets partially overlap."""
    left_df = pd.DataFrame({"openalex_id": ["W1", "W2", "W3"], "x": [1, 2, 3]})
    right_df = pd.DataFrame({"openalex_id": ["W2", "W3", "W4"], "x": [2, 3, 4]})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        left_path = path / "left.parquet"
        right_path = path / "right.parquet"
        left_df.to_parquet(left_path, index=False)
        right_df.to_parquet(right_path, index=False)
        report = compute_overlap_report(left_path, right_path)
    assert report.get("error") is None
    assert report["size_left"] == 3
    assert report["size_right"] == 3
    assert report["overlap_count"] == 2
    assert report["identical"] is False
    assert report["only_in_left_count"] == 1
    assert report["only_in_right_count"] == 1
    assert "W1" in report["sample_only_in_left_ids"]
    assert "W4" in report["sample_only_in_right_ids"]


def test_run_overlap_audit_writes_json_and_md() -> None:
    """run_overlap_audit writes JSON and Markdown report to out_dir."""
    df = pd.DataFrame({"openalex_id": ["A", "B"], "x": [1, 2]})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        left_path = path / "l.parquet"
        right_path = path / "r.parquet"
        df.to_parquet(left_path, index=False)
        df.to_parquet(right_path, index=False)
        out_dir = path / "reports"
        report, json_path, md_path = run_overlap_audit(left_path, right_path, out_dir, label_left="L", label_right="R")
        assert json_path.exists()
        assert md_path.exists()
        assert report["identical"] is True
        assert "l_vs_r" in json_path.name.lower() or "L_vs_R" in json_path.name
