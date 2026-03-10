"""Tests for target-level reporting: target profile, eligibility, empty/missing counts_by_year."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scholarly_outcome_prediction.features.targets import (
    compute_target_construction_diagnostics,
    _is_empty_or_missing_counts_by_year,
    prepare_df_for_target,
)
from scholarly_outcome_prediction.diagnostics.target_profile import (
    build_target_profile,
    build_target_semantics_description,
    write_target_profile,
    write_target_profile_md,
)


def test_is_empty_or_missing_counts_by_year() -> None:
    """Null, NaN, empty string, or parse-to-empty -> True."""
    assert _is_empty_or_missing_counts_by_year(None) is True
    assert _is_empty_or_missing_counts_by_year(np.nan) is True
    assert _is_empty_or_missing_counts_by_year("") is True
    assert _is_empty_or_missing_counts_by_year("[]") is True
    assert _is_empty_or_missing_counts_by_year("[{}]") is True
    assert _is_empty_or_missing_counts_by_year(json.dumps([{"year": 2019, "cited_by_count": 1}])) is False


def test_compute_target_construction_diagnostics() -> None:
    """Empty/missing counts_by_year and cited_by_count zero vs positive are counted correctly."""
    df = pd.DataFrame({
        "counts_by_year_json": [
            None,
            "[]",
            json.dumps([{"year": 2019, "cited_by_count": 1}]),
            None,
        ],
        "cited_by_count": [0, 5, 10, 3],
    })
    diag = compute_target_construction_diagnostics(df)
    assert diag["n_rows_empty_or_missing_counts_by_year"] == 3
    assert diag["n_empty_counts_and_cited_by_count_zero"] == 1
    assert diag["n_empty_counts_but_cited_by_count_positive"] == 2


def test_compute_target_construction_diagnostics_no_cited_by_count() -> None:
    """When cited_by_count column is missing, zero/positive counts are None."""
    df = pd.DataFrame({"counts_by_year_json": [None, "[]"]})
    diag = compute_target_construction_diagnostics(df)
    assert diag["n_rows_empty_or_missing_counts_by_year"] == 2
    assert diag["n_empty_counts_and_cited_by_count_zero"] is None
    assert diag["n_empty_counts_but_cited_by_count_positive"] is None


def test_prepare_df_for_target_includes_construction_diagnostics() -> None:
    """prepare_df_for_target adds target_construction_diagnostics and eligibility to eligibility_info."""
    df = pd.DataFrame({
        "publication_year": [2017, 2018],
        "counts_by_year_json": [
            json.dumps([{"year": 2017, "cited_by_count": 1}, {"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 3}]),
            json.dumps([{"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 4}]),
        ],
        "cited_by_count": [6, 6],
    })
    out_df, info = prepare_df_for_target(
        df, target_name="y", target_mode="calendar_horizon",
        horizon_years=2, include_publication_year=True,
    )
    assert "target_construction_diagnostics" in info
    assert "n_eligible" in info
    assert "n_excluded_horizon_incomplete" in info
    assert "eligibility_cutoff_description" in info
    assert "n_eligible_empty_counts_produced_zero_target" in info


def test_build_target_profile_synthetic_horizon() -> None:
    """Target profile for a small synthetic horizon target has required keys and zero-rate."""
    eligibility_info = {
        "target_mode": "calendar_horizon",
        "n_rows_raw": 100,
        "n_eligible": 80,
        "n_excluded_horizon_incomplete": 20,
        "max_available_citation_year": 2022,
        "horizon_years": 2,
        "include_publication_year": True,
        "eligibility_cutoff_description": "Row eligible iff publication_year + horizon_years <= 2022.",
        "target_construction_diagnostics": {
            "n_rows_empty_or_missing_counts_by_year": 10,
            "n_empty_counts_and_cited_by_count_zero": 5,
            "n_empty_counts_but_cited_by_count_positive": 5,
        },
        "n_eligible_empty_counts_produced_zero_target": 3,
    }
    target_config = {
        "name": "citations_within_2_calendar_years",
        "target_mode": "calendar_horizon",
        "source": "counts_by_year",
        "horizon_years": 2,
        "include_publication_year": True,
        "transform": "log1p",
    }
    untransformed = pd.Series([0, 0, 1, 2, 5, 10, 0, 3])
    transformed = pd.Series([0.0, 0.0, np.log1p(1), np.log1p(2), np.log1p(5), np.log1p(10), 0.0, np.log1p(3)])
    profile = build_target_profile(
        eligibility_info,
        target_config,
        untransformed,
        transformed_target_series=transformed,
        run_id="test-run",
        dataset_id="test-dataset",
        experiment_name="test_exp",
    )
    assert profile["report_name"] == "target_profile"
    assert profile["target_config_summary"]["target_name"] == "citations_within_2_calendar_years"
    assert profile["target_config_summary"]["target_mode"] == "calendar_horizon"
    assert profile["eligibility_summary"]["n_eligible_for_target"] == 80
    assert profile["eligibility_summary"]["n_excluded_horizon_incomplete"] == 20
    unt = profile["untransformed_target_distribution"]
    assert unt["zero_target_rate"] == 0.375
    assert unt["nonzero_target_rate"] == 0.625
    assert "count" in unt and unt["count"] == 8
    assert profile["empty_missing_counts_diagnostics"]["n_rows_empty_or_missing_counts_by_year"] == 10
    assert profile["empty_missing_counts_diagnostics"]["n_empty_counts_but_cited_by_count_positive"] == 5
    assert profile["transformed_target_distribution"] is not None
    assert "counts_by_year_handling_note" in profile
    assert "note_dataset_vs_target" in profile


def test_target_profile_zero_rate_computed_correctly() -> None:
    """Zero-target rate and nonzero rate sum to 1 and reflect series."""
    eligibility_info = {"target_mode": "calendar_horizon", "n_rows_raw": 5, "n_eligible": 5, "n_excluded_horizon_incomplete": 0}
    target_config = {"name": "y", "target_mode": "calendar_horizon"}
    ser = pd.Series([0, 0, 0, 1, 2])
    profile = build_target_profile(eligibility_info, target_config, ser, None)
    assert profile["untransformed_target_distribution"]["zero_target_rate"] == 0.6
    assert profile["untransformed_target_distribution"]["nonzero_target_rate"] == 0.4


def test_target_profile_distinct_from_dataset_profile() -> None:
    """Target profile has report_name target_profile and describes target, not raw dataset."""
    eligibility_info = {"target_mode": "calendar_horizon", "n_rows_raw": 10, "n_eligible": 8}
    profile = build_target_profile(eligibility_info, {"name": "y", "target_mode": "calendar_horizon"}, pd.Series([1, 2, 3]), None)
    assert profile["report_name"] == "target_profile"
    assert "untransformed_target_distribution" in profile
    assert "eligibility_summary" in profile
    assert "citation_distribution" not in profile
    assert "missingness" not in profile


def test_build_target_semantics_description_calendar_horizon() -> None:
    """Semantics description includes horizon and eligibility counts when provided."""
    cfg = {"target_mode": "calendar_horizon", "horizon_years": 2, "include_publication_year": True, "transform": "log1p", "name": "y"}
    desc = build_target_semantics_description(cfg, {"n_eligible": 80, "n_excluded_horizon_incomplete": 20})
    assert "publication_year" in desc
    assert "80" in desc
    assert "20" in desc
    assert "log1p" in desc


def test_write_target_profile_json_and_md(tmp_path: Path) -> None:
    """write_target_profile and write_target_profile_md produce files with expected content."""
    profile = {
        "report_name": "target_profile",
        "target_config_summary": {"target_name": "y", "target_mode": "calendar_horizon"},
        "eligibility_summary": {"n_eligible": 5},
        "untransformed_target_distribution": {"zero_target_rate": 0.2},
        "note_dataset_vs_target": "Different concepts.",
    }
    json_path = tmp_path / "target_profile.json"
    md_path = tmp_path / "target_profile.md"
    write_target_profile(profile, json_path)
    write_target_profile_md(profile, md_path)
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["report_name"] == "target_profile"
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "Target profile" in md_text
    assert "Different concepts" in md_text
