"""Tests for calendar-horizon target construction and eligibility (no live API)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scholarly_outcome_prediction.features.targets import (
    compute_calendar_horizon_target,
    compute_max_available_citation_year,
    is_horizon_eligible,
    build_calendar_horizon_target_column,
    prepare_df_for_target,
)
from scholarly_outcome_prediction.features.targets import _parse_counts_by_year


def test_compute_calendar_horizon_include_publication_year() -> None:
    """With include_publication_year=True, horizon_years=2, sum Y, Y+1, Y+2."""
    counts = {2018: 1, 2019: 2, 2020: 3}
    t = compute_calendar_horizon_target(2018, counts, horizon_years=2, include_publication_year=True)
    assert t == 1 + 2 + 3


def test_compute_calendar_horizon_exclude_publication_year() -> None:
    """With include_publication_year=False, horizon_years=2, sum Y+1, Y+2."""
    counts = {2018: 1, 2019: 2, 2020: 3}
    t = compute_calendar_horizon_target(2018, counts, horizon_years=2, include_publication_year=False)
    assert t == 2 + 3


def test_compute_calendar_horizon_missing_years_contribute_zero() -> None:
    """Years absent from counts_by_year contribute 0."""
    counts = {2018: 10}
    t = compute_calendar_horizon_target(2018, counts, horizon_years=2, include_publication_year=True)
    assert t == 10


def test_compute_calendar_horizon_none_publication_year() -> None:
    """None publication_year returns None."""
    assert compute_calendar_horizon_target(None, {2018: 1}, 2, True) is None


def test_compute_calendar_horizon_empty_counts() -> None:
    """Empty counts returns 0 sum (null/empty explicitly treated as zero yearly-count series)."""
    t = compute_calendar_horizon_target(2018, {}, horizon_years=2, include_publication_year=True)
    assert t == 0


def test_parse_counts_by_year_null_maps_to_zero_series() -> None:
    """Null or NaN counts_by_year_json is explicitly mapped to {} (zero yearly-count series)."""
    assert _parse_counts_by_year(None) == {}
    assert _parse_counts_by_year(np.nan) == {}
    # Empty dict then yields target 0 (documented assumption)
    assert compute_calendar_horizon_target(2019, _parse_counts_by_year(None), 2, True) == 0


def test_h2_include_publication_year_sums_three_years() -> None:
    """Evidence: horizon_years=2 + include_publication_year=True sums 3 years (Y, Y+1, Y+2), not 2.
    See docs/calendar_horizon_semantics_diagnosis.md."""
    publication_year = 2018
    counts = {2018: 1, 2019: 2, 2020: 3}
    t = compute_calendar_horizon_target(publication_year, counts, horizon_years=2, include_publication_year=True)
    assert t == 1 + 2 + 3
    # So years summed are 2018, 2019, 2020 (3 years). Eligibility must require data through 2020.
    assert is_horizon_eligible(2018, 2020, 2, True) is True
    assert is_horizon_eligible(2018, 2019, 2, True) is False


def test_is_horizon_eligible() -> None:
    """Eligible when last_year_needed (pub_year + horizon_years) <= max_available_citation_year."""
    assert is_horizon_eligible(2018, 2020, 2, True) is True   # need through 2020
    assert is_horizon_eligible(2019, 2020, 2, True) is False   # need through 2021
    assert is_horizon_eligible(2018, 2020, 2, False) is True   # need 2019,2020; last 2020
    assert is_horizon_eligible(2019, 2020, 2, False) is False  # need 2020,2021; max 2020 -> ineligible
    assert is_horizon_eligible(2018, 2021, 2, False) is True   # need 2019,2020


def test_build_calendar_horizon_target_column() -> None:
    """Target column and eligibility from counts_by_year_json."""
    df = pd.DataFrame({
        "publication_year": [2017, 2018, 2019],
        "counts_by_year_json": [
            json.dumps([{"year": 2017, "cited_by_count": 1}, {"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 3}]),
            json.dumps([{"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 4}]),
            json.dumps([{"year": 2019, "cited_by_count": 5}]),
        ],
    })
    target, eligible, n_eligible, n_excluded = build_calendar_horizon_target_column(
        df, horizon_years=2, include_publication_year=True, target_name="y"
    )
    assert target.iloc[0] == 1 + 2 + 3
    assert target.iloc[1] == 2 + 4
    assert target.iloc[2] == 5
    assert n_eligible + n_excluded == 3
    assert eligible.iloc[0] == True
    assert eligible.iloc[2] == False


def test_prepare_df_for_target_proxy_passthrough() -> None:
    """Proxy mode returns df copy and empty eligibility_info."""
    df = pd.DataFrame({"a": [1, 2], "cited_by_count": [10, 20]})
    out, info = prepare_df_for_target(df, "cited_by_count", "proxy")
    assert len(out) == 2
    assert info["target_mode"] == "proxy"
    assert "n_eligible" not in info


def test_prepare_df_for_target_calendar_horizon_filters() -> None:
    """Calendar_horizon filters to eligible rows and adds target column."""
    df = pd.DataFrame({
        "publication_year": [2017, 2018],
        "counts_by_year_json": [
            json.dumps([{"year": 2017, "cited_by_count": 1}, {"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 3}]),
            json.dumps([{"year": 2018, "cited_by_count": 2}, {"year": 2019, "cited_by_count": 4}]),
        ],
    })
    out, info = prepare_df_for_target(
        df, "citations_within_2_calendar_years", "calendar_horizon",
        horizon_years=2, include_publication_year=True,
    )
    assert info["target_mode"] == "calendar_horizon"
    assert "n_eligible" in info
    assert "n_excluded_horizon_incomplete" in info
    assert "max_available_citation_year" in info
    assert "citations_within_2_calendar_years" in out.columns


def test_compute_max_available_citation_year() -> None:
    """Max year across all rows' counts_by_year."""
    df = pd.DataFrame({
        "counts_by_year_json": [
            json.dumps([{"year": 2019, "cited_by_count": 1}]),
            json.dumps([{"year": 2021, "cited_by_count": 2}]),
        ],
    })
    assert compute_max_available_citation_year(df) == 2021
