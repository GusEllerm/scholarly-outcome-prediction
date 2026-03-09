"""Component inventory: major modules and their role; critical path for run."""

from __future__ import annotations

from typing import Any

from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata


def build_component_inventory() -> dict[str, Any]:
    """Design-scoped list of major modules and critical path for `run`."""
    meta = report_metadata(report_scope="design", report_name="component_inventory", config_paths=None)
    return {
        **meta,
        "critical_path_for_run": [
            "scholarly_outcome_prediction.cli (run -> run_pipeline_from_configs)",
            "scholarly_outcome_prediction.settings (load_data_config, load_experiment_config)",
            "scholarly_outcome_prediction.acquisition.fetch (fetch_and_save)",
            "scholarly_outcome_prediction.acquisition.openalex_client (fetch_works_sample, fetch_works_page)",
            "scholarly_outcome_prediction.utils.io (load_jsonl, write_parquet, read_parquet)",
            "scholarly_outcome_prediction.data.normalize (normalize_works_to_dataframe)",
            "scholarly_outcome_prediction.features.build_features (build_feature_matrix)",
            "scholarly_outcome_prediction.data.split (train_test_split_df)",
            "scholarly_outcome_prediction.features.preprocess (build_preprocessor)",
            "scholarly_outcome_prediction.models.registry (get_model_builder)",
            "scholarly_outcome_prediction.models.baseline (BaselineRegressor)",
            "scholarly_outcome_prediction.models.xgboost_model (build_xgboost_regressor)",
            "scholarly_outcome_prediction.evaluation.report (save_model_pipeline, build_run_metadata, save_metrics)",
            "scholarly_outcome_prediction.evaluation.metrics (compute_metrics)",
            "scholarly_outcome_prediction.utils.seeds (set_global_seed)",
        ],
        "modules_by_domain": {
            "cli": ["scholarly_outcome_prediction.cli"],
            "config": ["scholarly_outcome_prediction.settings"],
            "acquisition": ["scholarly_outcome_prediction.acquisition.fetch", "scholarly_outcome_prediction.acquisition.openalex_client"],
            "data": ["scholarly_outcome_prediction.data.schemas", "scholarly_outcome_prediction.data.normalize", "scholarly_outcome_prediction.data.split"],
            "features": ["scholarly_outcome_prediction.features.build_features", "scholarly_outcome_prediction.features.preprocess"],
            "models": ["scholarly_outcome_prediction.models.registry", "scholarly_outcome_prediction.models.baseline", "scholarly_outcome_prediction.models.median_baseline", "scholarly_outcome_prediction.models.ridge_model", "scholarly_outcome_prediction.models.xgboost_model"],
            "evaluation": ["scholarly_outcome_prediction.evaluation.metrics", "scholarly_outcome_prediction.evaluation.report"],
            "utils": ["scholarly_outcome_prediction.utils.io", "scholarly_outcome_prediction.utils.seeds", "scholarly_outcome_prediction.logging_utils"],
        },
        "not_on_critical_path_for_run": [
            "scholarly_outcome_prediction.diagnostics (diagnostics only)",
            "scripts.run_experiment (alternative entrypoint)",
        ],
    }
