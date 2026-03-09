"""Tests for dataset identity propagation in run artifacts."""

import json
import tempfile
from pathlib import Path

from scholarly_outcome_prediction.evaluation import (
    build_run_metadata,
    save_metrics,
)
from scholarly_outcome_prediction.utils.io import load_json


def test_build_run_metadata_includes_effective_and_config_paths() -> None:
    """build_run_metadata persists effective_dataset_id, effective_processed_path, config paths."""
    meta = build_run_metadata(
        experiment_name="test_exp",
        target_name="y",
        target_transform="log1p",
        target_mode="proxy",
        model_name="baseline",
        model_params={},
        feature_numeric=[],
        feature_categorical=[],
        split_kind="random",
        split_test_size=0.2,
        split_random_state=42,
        train_size=80,
        test_size=20,
        effective_dataset_id="openalex_pilot_articles",
        effective_processed_path="data/processed/openalex_pilot_articles.parquet",
        data_config_path="configs/data/openalex_pilot_articles.yaml",
        experiment_config_path="configs/experiments/baseline_regression_time.yaml",
    )
    assert meta["effective_dataset_id"] == "openalex_pilot_articles"
    assert meta["effective_processed_path"] == "data/processed/openalex_pilot_articles.parquet"
    assert meta["data_config_path"] == "configs/data/openalex_pilot_articles.yaml"
    assert meta["experiment_config_path"] == "configs/experiments/baseline_regression_time.yaml"
    assert meta["dataset_id"] == "openalex_pilot_articles"
    assert "run_id" in meta


def test_save_metrics_persists_run_metadata() -> None:
    """Saved metrics JSON contains run_metadata keys including effective_* and config paths."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "metrics.json"
        save_metrics(
            {"rmse": 1.0, "mae": 0.5},
            path,
            run_metadata=build_run_metadata(
                experiment_name="e",
                target_name="y",
                target_transform=None,
                target_mode="proxy",
                model_name="m",
                model_params={},
                feature_numeric=[],
                feature_categorical=[],
                split_kind="time",
                split_test_size=0.2,
                split_random_state=42,
                train_size=80,
                test_size=20,
                effective_dataset_id="ds1",
                effective_processed_path="/path/to/ds1.parquet",
                data_config_path="configs/data/ds1.yaml",
                experiment_config_path="configs/experiments/exp1.yaml",
            ),
        )
        out = load_json(path)
        assert out["effective_dataset_id"] == "ds1"
        assert out["effective_processed_path"] == "/path/to/ds1.parquet"
        assert out["data_config_path"] == "configs/data/ds1.yaml"
        assert out["experiment_config_path"] == "configs/experiments/exp1.yaml"
        assert out["rmse"] == 1.0


def test_run_metadata_includes_validation_and_split_params() -> None:
    """Run metadata can store validation_summary_path and temporal split params."""
    from scholarly_outcome_prediction.evaluation import build_run_metadata

    meta = build_run_metadata(
        experiment_name="e",
        target_name="y",
        target_transform=None,
        target_mode="proxy",
        model_name="m",
        model_params={},
        feature_numeric=[],
        feature_categorical=[],
        split_kind="time",
        split_test_size=0.2,
        split_random_state=42,
        train_size=80,
        test_size=20,
        validation_summary_path="artifacts/reports/foo_dataset_validation.json",
        train_year_end=2018,
        test_year_start=2019,
    )
    assert meta["validation_summary_path"] == "artifacts/reports/foo_dataset_validation.json"
    assert meta["train_year_end"] == 2018
    assert meta["test_year_start"] == 2019


def test_run_metadata_dataset_mode() -> None:
    """Run metadata can store dataset_mode (representative vs temporal)."""
    from scholarly_outcome_prediction.evaluation import build_run_metadata

    meta = build_run_metadata(
        experiment_name="e",
        target_name="cited_by_count",
        target_transform="log1p",
        target_mode="proxy",
        model_name="m",
        model_params={},
        feature_numeric=[],
        feature_categorical=[],
        split_kind="random",
        split_test_size=0.2,
        split_random_state=42,
        train_size=80,
        test_size=20,
        dataset_mode="representative",
    )
    assert meta["dataset_mode"] == "representative"
    assert meta["target_mode"] == "proxy"
