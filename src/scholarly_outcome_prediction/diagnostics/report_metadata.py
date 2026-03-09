"""Standard report metadata: scope, name, timestamps, provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def report_metadata(
    report_scope: str,
    report_name: str,
    *,
    dataset_id: str | None = None,
    run_id: str | None = None,
    report_id: str | None = None,
    audit_id: str | None = None,
    source_dataset_path: str | None = None,
    source_dataset_id: str | None = None,
    config_paths: dict[str, str] | list[str] | None = None,
) -> dict[str, Any]:
    """
    Standard header for diagnostic/report artifacts.
    Scopes: run | dataset | design | experiment

    Use run_id only when the value identifies a specific execution instance of a run.
    Use report_id or audit_id for dataset-scoped or design-scoped reports (e.g. validation
    report identity, artifact audit identity). Use source_dataset_id for the dataset
    the report is about when that differs from report_id/audit_id.
    """
    meta: dict[str, Any] = {
        "report_scope": report_scope,
        "report_name": report_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if dataset_id is not None:
        meta["dataset_id"] = dataset_id
    if run_id is not None:
        meta["run_id"] = run_id
    if report_id is not None:
        meta["report_id"] = report_id
    if audit_id is not None:
        meta["audit_id"] = audit_id
    if source_dataset_path is not None:
        meta["source_dataset_path"] = source_dataset_path
    if source_dataset_id is not None:
        meta["source_dataset_id"] = source_dataset_id
    if config_paths is not None:
        meta["config_paths"] = config_paths
    return meta
