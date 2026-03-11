"""Diagnostics and transparency utilities for pipeline inspection."""

from scholarly_outcome_prediction.diagnostics.dataset_profile import profile_dataset
from scholarly_outcome_prediction.diagnostics.feature_usage import build_feature_usage_report
from scholarly_outcome_prediction.diagnostics.artifact_audit import audit_run_artifacts
from scholarly_outcome_prediction.diagnostics.preprocessing_audit import audit_preprocessing_and_leakage
from scholarly_outcome_prediction.diagnostics.pipeline_trace import (
    build_pipeline_trace,
    build_pipeline_trace_from_run_context,
)
from scholarly_outcome_prediction.diagnostics.component_inventory import build_component_inventory
from scholarly_outcome_prediction.diagnostics.dataset_overlap import (
    compute_overlap_report,
    run_overlap_audit,
)

__all__ = [
    "profile_dataset",
    "build_feature_usage_report",
    "audit_run_artifacts",
    "audit_preprocessing_and_leakage",
    "build_pipeline_trace",
    "build_pipeline_trace_from_run_context",
    "build_component_inventory",
    "compute_overlap_report",
    "run_overlap_audit",
]
