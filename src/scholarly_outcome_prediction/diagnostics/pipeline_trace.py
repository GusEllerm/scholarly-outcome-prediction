"""Pipeline trace: functional path for 'run' from CLI to artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata
from scholarly_outcome_prediction.utils.io import load_json


def build_pipeline_trace() -> dict[str, Any]:
    """
    Design-scoped static trace of the execution path for:
    scholarly-outcome-prediction run --data-config ... --baseline-config ... --xgb-config ...
    Use for architecture documentation. For run-specific facts use build_pipeline_trace_from_run_context.
    """
    meta = report_metadata(
        report_scope="design",
        report_name="pipeline_trace",
        config_paths=None,
    )
    return {
        **meta,
        "config_paths": None,
        "config_paths_note": "Design-scoped trace has no run context; config paths are not applicable. For actual paths use a run-scoped pipeline trace from the CLI run command.",
        "entrypoint": "scholarly_outcome_prediction.cli:app (typer), command: run",
        "run_handler": "run_pipeline_from_configs(data_config_path, baseline_config_path, xgb_config_path)",
        "steps": [
            {
                "step": 1,
                "name": "resolve_root",
                "code": "root = data_config_path.resolve().parents[2]",
            },
            {
                "step": 2,
                "name": "load_configs",
                "modules": ["scholarly_outcome_prediction.settings"],
                "calls": ["load_data_config(data_config_path)", "load_current_experiment_config(baseline_config_path)", "load_current_experiment_config(xgb_config_path)"],
                "outputs": ["data_cfg", "base_cfg", "xgb_cfg"],
            },
            {
                "step": 3,
                "name": "fetch",
                "modules": ["scholarly_outcome_prediction.acquisition.fetch", "scholarly_outcome_prediction.acquisition.openalex_client"],
                "calls": ["fetch_and_save(output_path, sample_size, from_publication_date, to_publication_date, seed, work_types, sort, stratify_by_year, use_random_sample)"],
                "reads": ["data_cfg.output_path", "data_cfg.sample_size", "data_cfg.from_publication_date", "data_cfg.to_publication_date", "data_cfg.seed", "data_cfg.work_types", "data_cfg.stratify_by_year", "data_cfg.use_random_sample"],
                "writes": ["root / data_cfg.output_path (JSONL)"],
            },
            {
                "step": 4,
                "name": "prepare",
                "modules": ["scholarly_outcome_prediction.utils.io", "scholarly_outcome_prediction.data.normalize"],
                "calls": ["load_jsonl(out_path)", "normalize_works_to_dataframe(records)", "write_parquet(df, processed_path)"],
                "reads": ["raw JSONL from step 3"],
                "writes": ["root/data/processed/{data_cfg.dataset_name}.parquet"],
            },
            {
                "step": 5,
                "name": "train_and_evaluate_per_experiment",
                "loop_over": ["base_cfg", "xgb_cfg"],
                "substeps": [
                    "set_global_seed(cfg.split.random_state)",
                    "read_parquet(processed_path)",
                    "build_feature_matrix(df, ...) -> X, y",
                    "concat(X,y), dropna(subset=[target])",
                    "train_test_split_df(full, ...) -> train_df, test_df",
                    "build_preprocessor(num_feat, cat_feat)",
                    "get_model_builder(cfg.model.name)(params=...)",
                    "Pipeline(preprocessor, model).fit(X_train, y_train)",
                    "save_model_pipeline(pipe, model_dir / {experiment_name}.joblib)",
                    "pipe.predict(X_test), compute_metrics(y_test, y_pred)",
                    "build_run_metadata(...), save_metrics(metrics, path, run_metadata)",
                ],
                "reads": ["processed_path (from step 4)", "cfg.features", "cfg.target", "cfg.split", "cfg.model", "cfg.evaluation"],
                "writes": ["artifacts/models/{experiment_name}.joblib", "artifacts/metrics/{experiment_name}.json"],
            },
        ],
        "config_keys_described": [
            "data (YAML): dataset_name, seed, sample_size, from_publication_date, to_publication_date, output_path, work_types, stratify_by_year, use_random_sample",
            "baseline experiment (YAML): experiment_name, task_type, data.processed_path, data.dataset_id, target.*, features.*, split.*, model.*, evaluation.*",
            "xgb experiment (YAML): same structure",
        ],
        "design_note": "This is a static architecture description, not a runtime trace. Config paths are not set. processed_path in experiment config may point to a different file than the one written if configs reference a different dataset_name.",
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key from config-like object."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump().get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key, default)
    return default if not isinstance(obj, dict) else obj.get(key, default)


def build_pipeline_trace_from_run_context(
    *,
    run_id: str,
    data_config_path: Path,
    data_cfg: Any,
    baseline_config_path: Path,
    base_cfg: Any,
    xgb_config_path: Path,
    xgb_cfg: Any,
    effective_processed_path: Path,
    validation_json_path: Path | None,
    stages_completed: dict[str, bool],
    metrics_paths: list[Path],
    model_paths: list[Path],
    dataset_id: str | None = None,
    target_profile_path: Path | None = None,
    target_eligibility_summary: dict[str, Any] | None = None,
    target_mode: str | None = None,
    target_source: str | None = None,
    horizon_years: int | None = None,
    include_publication_year: bool | None = None,
) -> dict[str, Any]:
    """
    Build a run-scoped pipeline trace with effective fetch controls, experiment details,
    stage completion, and consistency checks. run_id must identify this execution instance;
    dataset_id is the dataset name (defaults to run_id if not provided for backward compat).
    """
    effective_dataset_id = dataset_id if dataset_id is not None else run_id
    meta = report_metadata(
        report_scope="run",
        report_name="pipeline_trace",
        run_id=run_id,
        dataset_id=effective_dataset_id,
        source_dataset_path=str(effective_processed_path),
        source_dataset_id=effective_dataset_id,
        config_paths={
            "data": str(data_config_path.resolve()) if data_config_path else None,
            "baseline_experiment": str(baseline_config_path.resolve()) if baseline_config_path else None,
            "xgb_experiment": str(xgb_config_path.resolve()) if xgb_config_path else None,
        },
    )

    # Effective sampling strategy description
    stratify = _get(data_cfg, "stratify_by_year", False)
    use_random = _get(data_cfg, "use_random_sample", False)
    if stratify and use_random:
        effective_sampling = "stratified_by_year_with_random_sample"
    elif stratify:
        effective_sampling = "stratified_by_year_cursor_paging"
    else:
        effective_sampling = "single_slice_api_or_paging"

    data_block: dict[str, Any] = {
        "data_config_path": str(data_config_path.resolve()) if data_config_path else None,
        "configured_dataset_id": _get(data_cfg, "dataset_name"),
        "effective_dataset_id": effective_dataset_id,
        "raw_output_path": str(Path(_get(data_cfg, "output_path", ""))),
        "effective_processed_path": str(effective_processed_path),
        "year_range": {
            "from_publication_date": _get(data_cfg, "from_publication_date"),
            "to_publication_date": _get(data_cfg, "to_publication_date"),
        },
        "work_types": _get(data_cfg, "work_types"),
        "sample_size": _get(data_cfg, "sample_size"),
        "seed": _get(data_cfg, "seed"),
        "representative_vs_temporal": "representative" if "representative" in effective_dataset_id else ("temporal" if "temporal" in effective_dataset_id else "unspecified"),
        "stratify_by_year": stratify,
        "use_random_sample": use_random,
        "effective_sampling_strategy": effective_sampling,
    }

    def experiment_block(cfg: Any, config_path: Path) -> dict[str, Any]:
        data_sub = _get(cfg, "data")
        if hasattr(data_sub, "model_dump"):
            data_path = data_sub.model_dump()
        elif isinstance(data_sub, dict):
            data_path = data_sub
        else:
            data_path = {}
        model_sub = _get(cfg, "model")
        split_sub = _get(cfg, "split")
        target_sub = _get(cfg, "target")
        features_sub = _get(cfg, "features")
        return {
            "experiment_config_path": str(config_path),
            "experiment_name": _get(cfg, "experiment_name"),
            "model_name": _get(model_sub, "name") if model_sub is not None else None,
            "split_kind": _get(split_sub, "split_kind") if split_sub is not None else None,
            "target_name": _get(target_sub, "name") if target_sub is not None else None,
            "target_mode": _get(target_sub, "target_mode") if target_sub is not None else None,
            "target_transform": _get(target_sub, "transform") if target_sub is not None else None,
            "feature_numeric": _get(features_sub, "numeric") if features_sub is not None else [],
            "feature_categorical": _get(features_sub, "categorical") if features_sub is not None else [],
            "configured_processed_path": data_path.get("processed_path") if isinstance(data_path, dict) else None,
            "effective_processed_path_used": str(effective_processed_path),
        }

    experiments = [
        experiment_block(base_cfg, baseline_config_path),
        experiment_block(xgb_cfg, xgb_config_path),
    ]

    # Cross-checks
    data_config_dataset_id = _get(data_cfg, "dataset_name") or effective_dataset_id
    per_metrics_dataset_id: dict[str, Any] = {}
    all_agree = True
    for p in metrics_paths:
        if not p.exists():
            per_metrics_dataset_id[p.name] = {"error": "file not found", "agree": None}
            all_agree = False
            continue
        try:
            metrics_data = load_json(p)
            # Prefer effective_dataset_id (what was actually used); fall back to dataset_id
            metrics_id = metrics_data.get("effective_dataset_id") or metrics_data.get("dataset_id")
            agree = metrics_id is not None and metrics_id == data_config_dataset_id
            if not agree and metrics_id is not None:
                all_agree = False
            per_metrics_dataset_id[p.name] = {
                "metrics_effective_dataset_id": metrics_id,
                "agree": agree if metrics_id is not None else None,
            }
        except Exception as e:
            per_metrics_dataset_id[p.name] = {"error": str(e), "agree": None}
            all_agree = False

    cross_checks: dict[str, Any] = {
        "data_config_dataset_id_equals_metrics_dataset_id": {
            "data_config_dataset_id": data_config_dataset_id,
            "per_metrics_file": per_metrics_dataset_id,
            "all_agree": all_agree,
        },
        "experiment_processed_path_equals_effective": [
            (experiments[i].get("configured_processed_path"), experiments[i].get("effective_processed_path_used"))
            for i in range(len(experiments))
        ],
        "validation_path_used": str(validation_json_path) if validation_json_path else None,
        "model_artifacts_exist": [str(p) for p in model_paths if p.exists()],
        "model_artifacts_missing": [str(p) for p in model_paths if not p.exists()],
        "metrics_artifacts_exist": [str(p) for p in metrics_paths if p.exists()],
        "metrics_artifacts_missing": [str(p) for p in metrics_paths if not p.exists()],
    }

    # Machine-readable consistency checks (pass / fail / unknown)
    def _path_match(configured: str | None, effective: str) -> bool:
        if not configured:
            return True
        return Path(configured).resolve().name == Path(effective).resolve().name or configured == effective

    processed_path_matches = [
        _path_match(exp.get("configured_processed_path"), exp.get("effective_processed_path_used", ""))
        for exp in experiments
    ]
    consistency_checks: dict[str, str] = {
        "dataset_id_match": "pass" if all_agree and per_metrics_dataset_id else ("fail" if per_metrics_dataset_id and not all_agree else "unknown"),
        "processed_path_match": "pass" if all(processed_path_matches) else ("fail" if processed_path_matches else "unknown"),
        "validation_input_match": (
            "pass" if validation_json_path and validation_json_path.exists()
            else ("fail" if validation_json_path else "unknown")
        ),
        "artifacts_present": "pass" if (not any(not p.exists() for p in model_paths) and not any(not p.exists() for p in metrics_paths)) else "fail",
    }
    baseline_xgb_match = "unknown"
    if len(metrics_paths) >= 2:
        try:
            m0 = load_json(metrics_paths[0])
            m1 = load_json(metrics_paths[1])
            keys = ["effective_dataset_id", "effective_processed_path", "split_kind", "target_name", "target_mode", "target_transform"]
            vals0 = [m0.get(k) for k in keys]
            vals1 = [m1.get(k) for k in keys]
            baseline_xgb_match = "pass" if vals0 == vals1 else "fail"
        except Exception:
            baseline_xgb_match = "unknown"
    consistency_checks["baseline_xgb_metadata_match"] = baseline_xgb_match

    # Target-level block for calendar_horizon runs
    target_block: dict[str, Any] = {}
    if target_mode == "calendar_horizon":
        target_block = {
            "target_mode": target_mode,
            "target_source": target_source,
            "horizon_years": horizon_years,
            "include_publication_year": include_publication_year,
            "target_profile_path": str(target_profile_path) if target_profile_path else None,
            "eligibility_summary": {
                "n_rows_raw": target_eligibility_summary.get("n_rows_raw") if target_eligibility_summary else None,
                "n_eligible": target_eligibility_summary.get("n_eligible") if target_eligibility_summary else None,
                "n_excluded_horizon_incomplete": target_eligibility_summary.get("n_excluded_horizon_incomplete") if target_eligibility_summary else None,
                "max_available_citation_year": target_eligibility_summary.get("max_available_citation_year") if target_eligibility_summary else None,
                "eligibility_cutoff_description": target_eligibility_summary.get("eligibility_cutoff_description") if target_eligibility_summary else None,
            } if target_eligibility_summary else None,
        }

    return {
        **meta,
        "data_config": data_block,
        "experiments": experiments,
        "stages_completed": stages_completed,
        "consistency_checks": consistency_checks,
        "cross_checks": cross_checks,
        "metrics_paths": [str(p) for p in metrics_paths],
        "model_paths": [str(p) for p in model_paths],
        **({"target": target_block} if target_block else {}),
    }
