"""Audit run artifacts: presence, metadata completeness, baseline vs xgb consistency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata
from scholarly_outcome_prediction.utils.io import load_json


REQUIRED_METADATA_KEYS = [
    "experiment_name",
    "target_name",
    "target_transform",
    "target_mode",
    "model_name",
    "model_params",
    "feature_numeric",
    "feature_categorical",
    "split_kind",
    "split_test_size",
    "split_random_state",
    "train_size",
    "test_size",
    "dataset_id",
    "effective_dataset_id",
    "effective_processed_path",
]


def audit_run_artifacts(
    artifacts_root: Path,
    run_id: str | None = None,
    dataset_id: str | None = None,
    expected_metrics_names: list[str] | None = None,
    expected_model_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run-scoped audit of artifacts: metrics and model files expected vs found,
    required metadata keys, and baseline vs xgb agreement on dataset_id, processed_path,
    split_kind, target_name, target_mode, target_transform.
    When run_id/dataset_id or expected_* are not provided, scope is inferred (audit of whatever exists).
    """
    metrics_dir = artifacts_root / "metrics"
    models_dir = artifacts_root / "models"

    metrics_files = sorted(metrics_dir.glob("*.json")) if metrics_dir.exists() else []
    model_files = sorted(models_dir.glob("*.joblib")) if models_dir.exists() else []

    found_metrics_names = [p.stem for p in metrics_files]
    found_model_names = [p.stem for p in model_files]
    expected_metrics = expected_metrics_names or found_metrics_names
    expected_models = expected_model_names or found_model_names
    metrics_expected_not_found = [n for n in expected_metrics if n not in found_metrics_names]
    metrics_found = [n for n in expected_metrics if n in found_metrics_names]
    models_expected_not_found = [n for n in expected_models if n not in found_model_names]
    models_found = [n for n in expected_models if n in found_model_names]

    # run_id only when this audit is for a specific run instance; otherwise dataset-scoped with audit_id
    scope = "run" if run_id is not None else "dataset"
    meta = report_metadata(
        report_scope=scope,
        report_name="run_artifact_audit",
        run_id=run_id if scope == "run" else None,
        audit_id=(dataset_id or (run_id if run_id else "artifact_audit")),
        dataset_id=dataset_id,
        config_paths=None,
    )

    per_metric: dict[str, Any] = {}
    for path in metrics_files:
        try:
            data = load_json(path)
            present = [k for k in REQUIRED_METADATA_KEYS if k in data]
            missing = [k for k in REQUIRED_METADATA_KEYS if k not in data]
            has_metrics = any(k in data for k in ["rmse", "mae", "r2"])
            per_metric[path.name] = {
                "metadata_keys_present": present,
                "metadata_keys_missing": missing,
                "has_metric_values": has_metrics,
                "effective_dataset_id": data.get("effective_dataset_id"),
                "configured_dataset_id": data.get("dataset_id"),
                "effective_processed_path": data.get("effective_processed_path"),
                "experiment_config_path": data.get("experiment_config_path"),
                "train_size": data.get("train_size"),
                "test_size": data.get("test_size"),
                "target_name": data.get("target_name"),
                "target_mode": data.get("target_mode"),
                "target_transform": data.get("target_transform"),
                "split_kind": data.get("split_kind"),
            }
        except Exception as e:
            per_metric[path.name] = {"error": str(e)}

    # Baseline vs xgb agreement (when exactly two metrics files)
    baseline_xgb_agreement: dict[str, Any] = {}
    if len(metrics_files) >= 2:
        by_name = {p.stem: load_json(p) for p in metrics_files if p.exists()}
        keys_to_agree = ["effective_dataset_id", "effective_processed_path", "split_kind", "target_name", "target_mode", "target_transform"]
        names = list(by_name.keys())
        for key in keys_to_agree:
            vals = [by_name.get(n, {}).get(key) for n in names]
            baseline_xgb_agreement[key] = {
                "values": dict(zip(names, vals)),
                "agree": len(set(vals)) <= 1 if all(v is not None for v in vals) else None,
            }

    return {
        **meta,
        "artifacts_root": str(artifacts_root),
        "metrics_expected": expected_metrics,
        "metrics_found": metrics_found,
        "metrics_expected_not_found": metrics_expected_not_found,
        "model_expected": expected_models,
        "model_found": models_found,
        "model_expected_not_found": models_expected_not_found,
        "required_metadata_keys": REQUIRED_METADATA_KEYS,
        "per_metrics_file": per_metric,
        "baseline_xgb_agreement": baseline_xgb_agreement,
        "summary": {
            "total_metrics_jsons": len(metrics_files),
            "total_model_joblibs": len(model_files),
            "all_expected_metrics_found": len(metrics_expected_not_found) == 0,
            "all_expected_models_found": len(models_expected_not_found) == 0,
        },
    }
