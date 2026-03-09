"""Audit feature usage: normalized schema vs experiment configs vs leakage risk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scholarly_outcome_prediction.data.normalize import NORMALIZED_COLUMNS
from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata
from scholarly_outcome_prediction.utils.io import load_yaml


def build_feature_usage_report(
    configs_dir: Path,
    processed_path: Path | None = None,
) -> dict[str, Any]:
    """
    Compare normalized schema, feature columns in each experiment config,
    and optionally columns present in data. Report used, unused, missing, leakage-prone.
    """
    experiments_dir = configs_dir / "experiments"
    if not experiments_dir.exists():
        meta = report_metadata("design", "feature_usage_report", config_paths=[str(configs_dir)])
        return {**meta, "error": f"Experiments config dir not found: {experiments_dir}"}

    scope = "dataset" if (processed_path and processed_path.exists()) else "design"
    dataset_id = processed_path.stem if (processed_path and processed_path.exists()) else None
    meta = report_metadata(
        report_scope=scope,
        report_name="feature_usage_report",
        dataset_id=dataset_id,
        source_dataset_path=str(processed_path) if processed_path else None,
        config_paths=[str(experiments_dir)],
    )

    normalized_set = set(NORMALIZED_COLUMNS)
    # Columns that could leak target or future info if used as features
    leakage_risky = {"cited_by_count", "title", "openalex_id", "publication_date"}
    # Text/optional fields not used in current training
    optional_text = {"abstract_text", "fulltext_text", "has_abstract", "has_fulltext", "fulltext_origin"}

    experiment_configs: list[str] = []
    for p in sorted(experiments_dir.glob("*.yaml")):
        experiment_configs.append(p.name)

    data_columns: set[str] = set()
    if processed_path and processed_path.exists():
        data_columns = set(pd.read_parquet(processed_path).columns)

    per_experiment: dict[str, Any] = {}
    all_numeric: set[str] = set()
    all_categorical: set[str] = set()

    for name in experiment_configs:
        path = experiments_dir / name
        raw = load_yaml(path)
        num = raw.get("features", {}).get("numeric", [])
        cat = raw.get("features", {}).get("categorical", [])
        used = set(num) | set(cat)
        all_numeric.update(num)
        all_categorical.update(cat)
        missing_from_data = [c for c in used if data_columns and c not in data_columns]
        used_and_risky = used & leakage_risky
        per_experiment[name] = {
            "numeric": num,
            "categorical": cat,
            "used_columns": list(used),
            "configured_but_missing_from_data": missing_from_data,
            "used_and_leakage_risky": list(used_and_risky),
        }

    used_all = all_numeric | all_categorical
    unused_in_schema = normalized_set - used_all - {"cited_by_count"}  # target not a feature
    unused_in_schema = unused_in_schema - optional_text  # optional text not used yet
    available_not_used = normalized_set - used_all - optional_text - {"cited_by_count"}

    return {
        **meta,
        "normalized_schema_columns": list(NORMALIZED_COLUMNS),
        "optional_text_columns": list(optional_text),
        "leakage_risky_columns": list(leakage_risky),
        "all_numeric_used_any_config": list(all_numeric),
        "all_categorical_used_any_config": list(all_categorical),
        "per_experiment": per_experiment,
        "unused_metadata_columns": list(unused_in_schema),
        "available_not_used": list(available_not_used),
        "text_used": False,
    }


