"""
Canonical dataset statistics: single computation used by profile and validation.
Prevents drift between dataset_profile.json and validation JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compute_canonical_dataset_stats(df: pd.DataFrame, source_path: str | Path | None = None) -> dict[str, Any]:
    """
    Compute canonical stats from a processed DataFrame. Used by both profile and validation.
    Returns a single structure with: row_count, publication_year, type_counts, citation_distribution,
    missingness_summary, distinct counts, venue/topic/language non-null rates.
    """
    path_str = str(source_path) if source_path else None
    n = len(df)
    stats: dict[str, Any] = {
        "row_count": n,
        "source_path": path_str,
        "column_names": list(df.columns),
    }

    # Publication year
    year_col = "publication_year"
    if year_col in df.columns:
        valid = df[year_col].dropna()
        if len(valid):
            stats["publication_year"] = {
                "min": int(valid.min()),
                "max": int(valid.max()),
                "n_unique": int(valid.nunique()),
                "counts": df[year_col].value_counts().sort_index().astype(int).to_dict(),
            }
        else:
            stats["publication_year"] = {"min": None, "max": None, "n_unique": 0, "counts": {}}
    else:
        stats["publication_year"] = {}

    # Type counts
    if "type" in df.columns:
        stats["type_counts"] = df["type"].value_counts().astype(int).to_dict()
    else:
        stats["type_counts"] = {}

    # Citation distribution (cited_by_count)
    if "cited_by_count" in df.columns:
        ser = df["cited_by_count"].dropna().astype(float)
        if len(ser):
            stats["citation_distribution"] = {
                "min": float(ser.min()),
                "q05": float(ser.quantile(0.05)),
                "q25": float(ser.quantile(0.25)),
                "median": float(ser.quantile(0.50)),
                "q75": float(ser.quantile(0.75)),
                "q95": float(ser.quantile(0.95)),
                "max": float(ser.max()),
                "mean": float(ser.mean()),
            }
        else:
            stats["citation_distribution"] = {}
    else:
        stats["citation_distribution"] = {}

    # Missingness
    missing = df.isna().sum()
    stats["missingness_summary"] = {
        col: {"count": int(missing[col]), "pct": round(100.0 * missing[col] / n, 2) if n else 0}
        for col in df.columns
    }

    # Distinct counts
    for col, key in [
        ("venue_name", "distinct_venue_count"),
        ("primary_topic", "distinct_primary_topic_count"),
        ("language", "distinct_language_count"),
    ]:
        stats[key] = int(df[col].dropna().nunique()) if col in df.columns else 0

    # Non-null rates
    for col, key in [
        ("venue_name", "venue_name_non_null_pct"),
        ("primary_topic", "primary_topic_non_null_pct"),
        ("authors_count", "authors_count_non_null_pct"),
        ("institutions_count", "institutions_count_non_null_pct"),
    ]:
        if col in df.columns:
            stats[key] = round(100.0 * df[col].notna().sum() / n, 2) if n else 0.0
        else:
            stats[key] = 0.0

    # Target (cited_by_count) summary for profile compatibility
    if "cited_by_count" in df.columns:
        ser = df["cited_by_count"].dropna().astype(float)
        stats["target_cited_by_count"] = {
            "min": float(ser.min()) if len(ser) else None,
            "max": float(ser.max()) if len(ser) else None,
            "mean": float(ser.mean()) if len(ser) else None,
            "median": float(ser.median()) if len(ser) else None,
            "count": int(ser.count()),
            "missing": int(df["cited_by_count"].isna().sum()),
        }
    else:
        stats["target_cited_by_count"] = {}

    return stats
