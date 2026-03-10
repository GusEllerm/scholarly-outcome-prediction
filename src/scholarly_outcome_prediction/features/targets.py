"""
Target construction: proxy (cited_by_count) and calendar_horizon (from counts_by_year).

Calendar-horizon targets sum citations over a fixed number of calendar years.
Semantics are explicit: e.g. citations_within_2_calendar_years (publication year + next 2).
Not exact month-level windows.

Null or missing counts_by_year_json is explicitly treated as a zero yearly-count series:
the horizon sum is 0 for such rows (documented in _parse_counts_by_year and
compute_calendar_horizon_target).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _parse_counts_by_year(counts_by_year_json: Any) -> dict[int, int]:
    """
    Parse counts_by_year from JSON string or None. Returns dict year -> cited_by_count.

    Null, NaN, missing, or malformed input is explicitly mapped to an empty dict {},
    i.e. a zero yearly-count series: no citations in any year. Downstream horizon sums
    then treat each year in the window as 0, so the target value for such rows is 0.
    """
    if counts_by_year_json is None or (isinstance(counts_by_year_json, float) and np.isnan(counts_by_year_json)):
        return {}
    if isinstance(counts_by_year_json, dict):
        return {int(k): int(v) for k, v in counts_by_year_json.items() if _safe_int(k) is not None and _safe_int(v) is not None}
    if isinstance(counts_by_year_json, str):
        if not counts_by_year_json.strip():
            return {}
        try:
            data = json.loads(counts_by_year_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    else:
        data = counts_by_year_json
    if not isinstance(data, list):
        return {}
    out: dict[int, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        cnt = item.get("cited_by_count")
        if year is not None and cnt is not None:
            yi = _safe_int(year)
            ci = _safe_int(cnt)
            if yi is not None and ci is not None and ci >= 0:
                out[yi] = out.get(yi, 0) + ci
    return out


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _is_empty_or_missing_counts_by_year(val: Any) -> bool:
    """True if counts_by_year value is null, NaN, empty string, or parses to empty dict."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    d = _parse_counts_by_year(val)
    return len(d) == 0


def compute_target_construction_diagnostics(
    df: pd.DataFrame,
    counts_by_year_col: str = "counts_by_year_json",
    cited_by_count_col: str = "cited_by_count",
) -> dict[str, Any]:
    """
    From the raw (pre-eligibility) DataFrame, count rows by counts_by_year and cited_by_count.

    Used for target-level reporting: surfaces how many rows have empty/missing counts_by_year,
    and of those how many have cited_by_count zero vs positive (empty counts_by_year is not
    equivalent to "no citations").
    """
    if counts_by_year_col not in df.columns:
        return {
            "n_rows_empty_or_missing_counts_by_year": 0,
            "n_empty_counts_and_cited_by_count_zero": 0,
            "n_empty_counts_but_cited_by_count_positive": 0,
        }
    empty_missing = df[counts_by_year_col].apply(_is_empty_or_missing_counts_by_year)
    n_empty = int(empty_missing.sum())
    if cited_by_count_col not in df.columns:
        return {
            "n_rows_empty_or_missing_counts_by_year": n_empty,
            "n_empty_counts_and_cited_by_count_zero": None,
            "n_empty_counts_but_cited_by_count_positive": None,
        }
    cited = df[cited_by_count_col]
    n_zero = int((empty_missing & (cited == 0)).sum())
    n_positive = int((empty_missing & (cited > 0)).sum())
    return {
        "n_rows_empty_or_missing_counts_by_year": n_empty,
        "n_empty_counts_and_cited_by_count_zero": n_zero,
        "n_empty_counts_but_cited_by_count_positive": n_positive,
    }


def compute_calendar_horizon_target(
    publication_year: int | None,
    counts_by_year: dict[int, int],
    horizon_years: int,
    include_publication_year: bool,
) -> int | None:
    """
    Sum citations over a calendar-year horizon.

    If include_publication_year is True and horizon_years is 2:
        years = [Y, Y+1, Y+2]
    If include_publication_year is False and horizon_years is 2:
        years = [Y+1, Y+2]

    counts_by_year null/empty is explicitly treated as a zero yearly-count series:
    each year in the window contributes 0, so the returned target is 0 (not None).
    Returns None only when publication_year is None or horizon_years < 1.
    Years absent from counts_by_year contribute 0 (partial data may undercount).
    """
    if publication_year is None or horizon_years < 1:
        return None
    start = publication_year if include_publication_year else publication_year + 1
    end = publication_year + horizon_years
    total = 0
    for y in range(start, end + 1):
        total += counts_by_year.get(y, 0)
    return total


def is_horizon_eligible(
    publication_year: int | None,
    max_available_citation_year: int,
    horizon_years: int,
    include_publication_year: bool,
) -> bool:
    """
    True iff we have citation data through the full horizon so the target is fully observed.

    The last year we need is: publication_year + horizon_years (if include_publication_year,
    we need Y, Y+1, ..., Y+horizon_years; last is Y+horizon_years).
    So we require: publication_year + horizon_years <= max_available_citation_year.
    """
    if publication_year is None or horizon_years < 1:
        return False
    last_year_needed = publication_year + horizon_years
    return last_year_needed <= max_available_citation_year


def compute_max_available_citation_year(df: pd.DataFrame, counts_by_year_col: str = "counts_by_year_json") -> int | None:
    """
    From a DataFrame with counts_by_year_json, return the maximum citation year present
    across all rows (the latest year we have any citation data for).
    """
    max_year: int | None = None
    for val in df[counts_by_year_col].dropna():
        by_year = _parse_counts_by_year(val)
        for y in by_year:
            if max_year is None or y > max_year:
                max_year = y
    return max_year


def build_calendar_horizon_target_column(
    df: pd.DataFrame,
    horizon_years: int,
    include_publication_year: bool,
    counts_by_year_col: str = "counts_by_year_json",
    publication_year_col: str = "publication_year",
    target_name: str = "citations_within_2_calendar_years",
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Add a calendar-horizon target column and an eligibility mask.

    Returns:
        target_series: series with same index as df; NaN where ineligible or missing data.
        eligible_mask: boolean series, True where row is eligible (full horizon observed).
        n_eligible: count of eligible rows.
        n_excluded: count excluded for incomplete horizon or missing data.
    """
    if counts_by_year_col not in df.columns or publication_year_col not in df.columns:
        target = pd.Series(index=df.index, dtype=float)
        target[:] = np.nan
        return target, pd.Series(False, index=df.index), 0, len(df)

    max_year = compute_max_available_citation_year(df, counts_by_year_col)
    if max_year is None:
        target = pd.Series(index=df.index, dtype=float)
        target[:] = np.nan
        return target, pd.Series(False, index=df.index), 0, len(df)

    targets = []
    eligible = []
    for idx, row in df.iterrows():
        pub_year = _safe_int(row.get(publication_year_col))
        counts = _parse_counts_by_year(row.get(counts_by_year_col))
        t = compute_calendar_horizon_target(pub_year, counts, horizon_years, include_publication_year)
        elig = is_horizon_eligible(pub_year, max_year, horizon_years, include_publication_year) and t is not None
        targets.append(float(t) if t is not None else np.nan)
        eligible.append(elig)
    target_series = pd.Series(targets, index=df.index)
    eligible_series = pd.Series(eligible, index=df.index)
    n_eligible = int(eligible_series.sum())
    n_excluded = len(df) - n_eligible
    return target_series, eligible_series, n_eligible, n_excluded


def prepare_df_for_target(
    df: pd.DataFrame,
    target_name: str,
    target_mode: str,
    horizon_years: int | None = None,
    include_publication_year: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare DataFrame for modeling: for proxy, return df as-is with empty eligibility_info.
    For calendar_horizon, compute target from counts_by_year, filter to eligible rows, add target column.

    Returns (df_for_modeling, eligibility_info). eligibility_info has keys like
    n_rows_raw, n_eligible, n_excluded_horizon_incomplete, max_available_citation_year, etc.
    """
    eligibility_info: dict[str, Any] = {"target_mode": target_mode, "n_rows_raw": len(df)}
    if target_mode != "calendar_horizon" or horizon_years is None or horizon_years < 1:
        return df.copy(), eligibility_info

    counts_by_year_col = "counts_by_year_json"
    target_series, eligible_series, n_eligible, n_excluded = build_calendar_horizon_target_column(
        df,
        horizon_years=horizon_years,
        include_publication_year=include_publication_year,
        target_name=target_name,
    )
    max_year = compute_max_available_citation_year(df)
    eligibility_info["n_eligible"] = n_eligible
    eligibility_info["n_excluded_horizon_incomplete"] = n_excluded
    eligibility_info["max_available_citation_year"] = max_year
    eligibility_info["horizon_years"] = horizon_years
    eligibility_info["include_publication_year"] = include_publication_year
    diagnostics = compute_target_construction_diagnostics(df, counts_by_year_col=counts_by_year_col)
    eligibility_info["target_construction_diagnostics"] = diagnostics
    # Among eligible rows, how many have target 0 because counts_by_year was empty/missing
    empty_missing_mask = df[counts_by_year_col].apply(_is_empty_or_missing_counts_by_year)
    eligible_with_zero_target = eligible_series & (target_series == 0)
    eligibility_info["n_eligible_empty_counts_produced_zero_target"] = int(
        (eligible_with_zero_target & empty_missing_mask).sum()
    )
    last_year_needed = "publication_year + horizon_years"
    eligibility_info["eligibility_cutoff_description"] = (
        f"Row eligible iff {last_year_needed} <= max_available_citation_year "
        f"(max_available_citation_year={max_year} from observed counts_by_year in this dataset)."
    )

    out = df.loc[eligible_series].copy()
    out[target_name] = target_series.loc[eligible_series].values
    return out, eligibility_info
