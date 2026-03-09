"""Profile normalized/processed dataset: counts, distributions, missingness. Uses canonical stats for alignment with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scholarly_outcome_prediction.data.normalize import NORMALIZED_COLUMNS
from scholarly_outcome_prediction.diagnostics.dataset_stats import compute_canonical_dataset_stats
from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata


def profile_dataset(
    processed_path: Path,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """
    Read processed Parquet, compute canonical stats, add profile-only fields (categorical_tops,
    numeric_degenerate). Includes report_scope=dataset and provenance.
    """
    if not processed_path.exists():
        meta = report_metadata("dataset", "dataset_profile", dataset_id=dataset_id or processed_path.stem)
        return {**meta, "error": f"Path not found: {processed_path}", "row_count": 0}

    df = pd.read_parquet(processed_path)
    stats = compute_canonical_dataset_stats(df, source_path=processed_path)
    effective_dataset_id = dataset_id or processed_path.stem
    meta = report_metadata(
        report_scope="dataset",
        report_name="dataset_profile",
        dataset_id=effective_dataset_id,
        source_dataset_path=str(processed_path),
    )

    # Profile-only: categorical tops
    cat_cols = ["type", "language", "venue_name", "primary_topic"]
    categorical_tops: dict[str, list[tuple[str, int]]] = {}
    for col in cat_cols:
        if col in df.columns:
            top = df[col].value_counts().head(15)
            categorical_tops[col] = [(str(k), int(v)) for k, v in top.items()]

    # Profile-only: degenerate numeric check
    numeric_feats = ["publication_year", "referenced_works_count", "authors_count", "institutions_count"]
    numeric_degenerate: dict[str, bool] = {}
    for col in numeric_feats:
        if col in df.columns:
            ser = df[col].dropna()
            numeric_degenerate[col] = ser.nunique() <= 1 if len(ser) else True

    return {
        **meta,
        **stats,
        "missingness": stats.get("missingness_summary", {}),  # backward compat key
        "categorical_tops": categorical_tops,
        "numeric_degenerate": numeric_degenerate,
        "normalized_schema_columns": list(NORMALIZED_COLUMNS),
    }


def write_missingness_csv(processed_path: Path, out_path: Path) -> None:
    """Write missingness summary as CSV from canonical stats (column, missing_count, pct_missing)."""
    if not processed_path.exists():
        return
    df = pd.read_parquet(processed_path)
    stats = compute_canonical_dataset_stats(df)
    summary = stats.get("missingness_summary", {})
    if not summary:
        return
    rows = [{"column": col, "missing_count": v["count"], "pct_missing": v["pct"]} for col, v in summary.items()]
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
