"""
Target-level profile: the actual supervised learning label used by an experiment.

Distinct from dataset_profile (which describes the processed dataset: cited_by_count,
feature missingness, venue/topic diversity). This report describes the modeling target:
eligibility filtering, target distribution, zero-rate, and how missing/empty counts_by_year
is handled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata
from scholarly_outcome_prediction.utils.io import save_json


def _target_distribution_summary(ser: pd.Series) -> dict[str, float | int]:
    """Quantiles and mean for a numeric series; dropna for stats."""
    valid = ser.dropna()
    if len(valid) == 0:
        return {"count": 0, "min": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None, "mean": None}
    return {
        "count": int(valid.count()),
        "min": float(valid.min()),
        "q05": float(valid.quantile(0.05)),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.quantile(0.50)),
        "q75": float(valid.quantile(0.75)),
        "q95": float(valid.quantile(0.95)),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def _zero_nonzero_rates(ser: pd.Series) -> tuple[float, float]:
    """(zero_rate, nonzero_rate) in [0,1]; based on non-null values only."""
    valid = ser.dropna()
    n = len(valid)
    if n == 0:
        return 0.0, 0.0
    n_zero = int((valid == 0).sum())
    return n_zero / n, (n - n_zero) / n


def build_target_semantics_description(cfg: Any, eligibility_info: dict[str, Any] | None = None) -> str:
    """One-line description of target computation for run metadata (calendar_horizon)."""
    c = _get_target_config_dict(cfg) if cfg is not None else {}
    mode = c.get("target_mode")
    if mode != "calendar_horizon":
        return f"Target mode: {mode}; target name: {c.get('name', 'target')}."
    inc = c.get("include_publication_year", True)
    h = c.get("horizon_years") or 0
    transform = c.get("transform") or "none"
    if inc:
        desc = (
            f"Target = sum of counts_by_year for years publication_year through publication_year+{h - 1}. "
            f"Transform: {transform}."
        )
    else:
        desc = (
            f"Target = sum of counts_by_year for next {h} calendar years after publication year. "
            f"Transform: {transform}."
        )
    if eligibility_info:
        desc += (
            f" Eligible: {eligibility_info.get('n_eligible', '?')} rows; "
            f"excluded (incomplete horizon): {eligibility_info.get('n_excluded_horizon_incomplete', '?')}."
        )
    return desc


def _get_target_config_dict(cfg: Any) -> dict[str, Any]:
    """Extract target config as dict from config object or dict."""
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if hasattr(cfg, "__dict__"):
        return {k: getattr(cfg, k, None) for k in ["name", "target_mode", "source", "horizon_years", "include_publication_year", "transform"]}
    return dict(cfg) if isinstance(cfg, dict) else {}


def build_target_profile(
    eligibility_info: dict[str, Any],
    target_config: Any,
    untransformed_target_series: pd.Series,
    transformed_target_series: pd.Series | None = None,
    *,
    run_id: str | None = None,
    dataset_id: str | None = None,
    experiment_name: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    """
    Build the target-level profile for calendar-horizon (or proxy) runs.

    eligibility_info: from prepare_df_for_target (n_eligible, n_excluded, diagnostics, etc.).
    target_config: experiment target config (object or dict) with name, mode, source, horizon_years, etc.
    untransformed_target_series: the raw target column (e.g. citation count sum) for eligible rows.
    transformed_target_series: optional transformed target (e.g. log1p) used in training.
    """
    cfg = _get_target_config_dict(target_config)
    name = target_name or cfg.get("name", "target")
    meta = report_metadata(
        report_scope="run",
        report_name="target_profile",
        run_id=run_id,
        dataset_id=dataset_id,
    )
    if experiment_name is not None:
        meta["experiment_name"] = experiment_name

    # Target configuration summary
    target_config_summary = {
        "target_name": name,
        "target_mode": cfg.get("target_mode"),
        "target_source": cfg.get("source"),
        "horizon_years": cfg.get("horizon_years"),
        "include_publication_year": cfg.get("include_publication_year"),
        "target_transform": cfg.get("transform"),
    }

    # Semantics note (calendar-horizon)
    target_semantics_note = None
    if cfg.get("target_mode") == "calendar_horizon":
        inc = cfg.get("include_publication_year", True)
        h = cfg.get("horizon_years") or 0
        if inc:
            target_semantics_note = (
                f"Target is the sum of citation counts from counts_by_year for years "
                f"publication_year through publication_year + {h - 1} (inclusive). "
                f"Calendar-year granularity, not exact month-level windows."
            )
        else:
            target_semantics_note = (
                f"Target is the sum of citation counts from counts_by_year for the next {h} "
                f"full calendar years after publication year (publication_year+1 through "
                f"publication_year+{h}). Calendar-year granularity."
            )

    # Eligibility summary
    eligibility_summary = {
        "n_rows_raw_processed_dataset": eligibility_info.get("n_rows_raw"),
        "n_eligible_for_target": eligibility_info.get("n_eligible"),
        "n_excluded_horizon_incomplete": eligibility_info.get("n_excluded_horizon_incomplete"),
        "max_available_citation_year": eligibility_info.get("max_available_citation_year"),
        "eligibility_cutoff_description": eligibility_info.get("eligibility_cutoff_description"),
    }

    # How missing/empty counts_by_year is handled (explicit)
    counts_by_year_handling_note = (
        "Missing or empty counts_by_year_json is explicitly treated as a zero yearly-count "
        "series: each year in the horizon contributes 0, so the target value for such rows is 0. "
        "This is documented in features/targets.py (_parse_counts_by_year, compute_calendar_horizon_target)."
    )

    # Diagnostics: empty/missing counts_by_year
    diagnostics = eligibility_info.get("target_construction_diagnostics") or {}
    empty_missing_diagnostics = {
        "n_rows_empty_or_missing_counts_by_year": diagnostics.get("n_rows_empty_or_missing_counts_by_year", 0),
        "n_empty_counts_and_cited_by_count_zero": diagnostics.get("n_empty_counts_and_cited_by_count_zero"),
        "n_empty_counts_but_cited_by_count_positive": diagnostics.get("n_empty_counts_but_cited_by_count_positive"),
        "n_eligible_empty_counts_produced_zero_target": eligibility_info.get("n_eligible_empty_counts_produced_zero_target"),
    }

    # Untransformed target distribution
    dist = _target_distribution_summary(untransformed_target_series)
    zero_rate, nonzero_rate = _zero_nonzero_rates(untransformed_target_series)
    untransformed = {
        **dist,
        "zero_target_rate": round(zero_rate, 4),
        "nonzero_target_rate": round(nonzero_rate, 4),
    }

    # Transformed target distribution (if provided)
    transformed_summary: dict[str, Any] = {}
    if transformed_target_series is not None and len(transformed_target_series.dropna()):
        transformed_summary = _target_distribution_summary(transformed_target_series)

    # Note: dataset cited_by_count vs modeling target
    note_dataset_vs_target = (
        "Dataset-level cited_by_count (in dataset_profile) is the current cumulative citation count "
        "from the snapshot. The calendar-horizon target is the sum of citations over a fixed number "
        "of calendar years from counts_by_year. They are different: do not confuse dataset citation "
        "distribution with the actual modeling target distribution reported here."
    )

    return {
        **meta,
        "target_config_summary": target_config_summary,
        "target_semantics_note": target_semantics_note,
        "eligibility_summary": eligibility_summary,
        "counts_by_year_handling_note": counts_by_year_handling_note,
        "empty_missing_counts_diagnostics": empty_missing_diagnostics,
        "untransformed_target_distribution": untransformed,
        "transformed_target_distribution": transformed_summary if transformed_summary else None,
        "note_dataset_vs_target": note_dataset_vs_target,
    }


def write_target_profile(profile: dict[str, Any], out_path: Path) -> None:
    """Write target profile as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(profile, out_path)


def write_target_profile_md(profile: dict[str, Any], out_path: Path) -> None:
    """Write a short markdown companion for the target profile."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Target profile",
        "",
        f"**Report scope:** {profile.get('report_scope', '')}",
        f"**Generated:** {profile.get('generated_at', '')}",
        "",
        "## Target configuration",
    ]
    for k, v in (profile.get("target_config_summary") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "## Eligibility",
    ])
    for k, v in (profile.get("eligibility_summary") or {}).items():
        lines.append(f"- {k}: {v}")
    if profile.get("target_semantics_note"):
        lines.extend(["", "## Target semantics", "", profile["target_semantics_note"]])
    if profile.get("counts_by_year_handling_note"):
        lines.extend(["", "## Missing/empty counts_by_year", "", profile["counts_by_year_handling_note"]])
    lines.extend(["", "## Empty/missing counts diagnostics"])
    for k, v in (profile.get("empty_missing_counts_diagnostics") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Untransformed target distribution"])
    for k, v in (profile.get("untransformed_target_distribution") or {}).items():
        lines.append(f"- {k}: {v}")
    if profile.get("transformed_target_distribution"):
        lines.extend(["", "## Transformed target distribution"])
        for k, v in profile["transformed_target_distribution"].items():
            lines.append(f"- {k}: {v}")
    lines.extend(["", "## Note", "", profile.get("note_dataset_vs_target", "")])
    out_path.write_text("\n".join(lines), encoding="utf-8")
