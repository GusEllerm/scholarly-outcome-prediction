"""Generate all diagnostic artifacts (profile, audit, design trace, etc.). Callable from CLI run or from scripts/generate_diagnostics.py."""

from __future__ import annotations

from pathlib import Path

from scholarly_outcome_prediction.diagnostics import (
    audit_preprocessing_and_leakage,
    audit_run_artifacts,
    build_component_inventory,
    build_feature_usage_report,
    build_pipeline_trace,
    profile_dataset,
)
from scholarly_outcome_prediction.diagnostics.dataset_profile import write_missingness_csv
from scholarly_outcome_prediction.utils.io import save_json


def generate_all_diagnostics(
    root: Path,
    processed_path: Path,
    dataset_id: str | None = None,
    *,
    out_dir: Path | None = None,
    configs_dir: Path | None = None,
    artifacts_root: Path | None = None,
    include_design_trace: bool = True,
) -> Path:
    """
    Write all diagnostic artifacts to out_dir (default root/artifacts/diagnostics).
    Does not write pipeline_trace.json (run-scoped); that is written by the CLI.
    When include_design_trace is True, writes pipeline_trace_design.json.
    Returns the output directory used.
    """
    out_dir = out_dir or root / "artifacts" / "diagnostics"
    configs_dir = configs_dir or root / "configs"
    artifacts_root = artifacts_root or root / "artifacts"
    effective_dataset_id = dataset_id or (processed_path.stem if processed_path else None)

    out_dir.mkdir(parents=True, exist_ok=True)

    inv = build_component_inventory()
    save_json(inv, out_dir / "component_inventory.json")

    if include_design_trace:
        trace = build_pipeline_trace()
        save_json(trace, out_dir / "pipeline_trace_design.json")

    profile = profile_dataset(processed_path, dataset_id=effective_dataset_id)
    profile["processed_path"] = str(processed_path)
    save_json(profile, out_dir / "dataset_profile.json")
    if processed_path.exists():
        write_missingness_csv(processed_path, out_dir / "missingness_summary.csv")

    feature_report = build_feature_usage_report(configs_dir, processed_path)
    save_json(feature_report, out_dir / "feature_usage_report.json")

    audit = audit_run_artifacts(artifacts_root, run_id=None, dataset_id=effective_dataset_id)
    save_json(audit, out_dir / "run_artifact_audit.json")

    preproc = audit_preprocessing_and_leakage()
    save_json(preproc, out_dir / "preprocessing_leakage_audit.json")

    return out_dir
