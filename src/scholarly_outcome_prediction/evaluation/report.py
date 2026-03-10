"""Save metrics and model artifacts with full run metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from scholarly_outcome_prediction.utils.io import save_json


def build_run_metadata(
    experiment_name: str,
    target_name: str,
    target_transform: str | None,
    target_mode: str,
    model_name: str,
    model_params: dict[str, Any],
    feature_numeric: list[str],
    feature_categorical: list[str],
    split_kind: str,
    split_test_size: float,
    split_random_state: int,
    train_size: int,
    test_size: int,
    dataset_id: str | None = None,
    run_id: str | None = None,
    effective_dataset_id: str | None = None,
    effective_processed_path: str | None = None,
    data_config_path: str | None = None,
    experiment_config_path: str | None = None,
    validation_summary_path: str | None = None,
    train_year_end: int | None = None,
    test_year_start: int | None = None,
    dataset_mode: str | None = None,
    target_source: str | None = None,
    horizon_years: int | None = None,
    include_publication_year: bool | None = None,
    target_eligibility: dict[str, Any] | None = None,
    target_semantics_description: str | None = None,
    target_zero_rate: float | None = None,
) -> dict[str, Any]:
    """Build a self-describing run metadata dict for artifact persistence."""
    meta = {
        "experiment_name": experiment_name,
        "target_name": target_name,
        "target_transform": target_transform,
        "target_mode": target_mode,
        "model_name": model_name,
        "model_params": model_params,
        "feature_numeric": feature_numeric,
        "feature_categorical": feature_categorical,
        "split_kind": split_kind,
        "split_test_size": split_test_size,
        "split_random_state": split_random_state,
        "train_size": train_size,
        "test_size": test_size,
        "dataset_id": effective_dataset_id or dataset_id,
        "effective_dataset_id": effective_dataset_id or dataset_id,
        "effective_processed_path": effective_processed_path,
        "data_config_path": data_config_path,
        "experiment_config_path": experiment_config_path,
        "run_id": run_id or datetime.now(timezone.utc).isoformat(),
    }
    if validation_summary_path is not None:
        meta["validation_summary_path"] = validation_summary_path
    if train_year_end is not None:
        meta["train_year_end"] = train_year_end
    if test_year_start is not None:
        meta["test_year_start"] = test_year_start
    if dataset_mode is not None:
        meta["dataset_mode"] = dataset_mode
    if target_source is not None:
        meta["target_source"] = target_source
    if horizon_years is not None:
        meta["horizon_years"] = horizon_years
    if include_publication_year is not None:
        meta["include_publication_year"] = include_publication_year
    if target_eligibility is not None:
        meta["target_eligibility"] = target_eligibility
    if target_semantics_description is not None:
        meta["target_semantics_description"] = target_semantics_description
    if target_zero_rate is not None:
        meta["target_zero_rate"] = target_zero_rate
    return meta


def save_metrics(metrics: dict[str, Any], path: Path, run_metadata: dict[str, Any] | None = None) -> None:
    """Write metrics to JSON; merge run_metadata so the artifact is self-describing."""
    out = dict(run_metadata or {})
    out.update(metrics)
    save_json(out, path)


def save_model_pipeline(pipe: Pipeline, path: Path) -> None:
    """Persist a fitted sklearn Pipeline (e.g. preprocessor + model) with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_model_pipeline(path: Path) -> Pipeline:
    """Load a persisted pipeline."""
    return joblib.load(path)
