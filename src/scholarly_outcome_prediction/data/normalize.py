"""Normalize raw OpenAlex works into a flat tabular DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scholarly_outcome_prediction.data.schemas import NormalizedWork


NORMALIZED_COLUMNS = [
    "openalex_id",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "cited_by_count",
    "referenced_works_count",
    "authors_count",
    "institutions_count",
    "venue_name",
    "open_access_is_oa",
    "primary_topic",
    "abstract_text",
    "fulltext_text",
    "has_abstract",
    "has_fulltext",
    "fulltext_origin",
]


def normalize_work(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw OpenAlex work to a flat dict; defensive around missing data."""
    n = NormalizedWork.from_openalex_work(raw)
    return n.model_dump()


def normalize_works_to_dataframe(raw_works: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of raw OpenAlex works to a DataFrame with normalized columns."""
    if not raw_works:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    rows = [normalize_work(w) for w in raw_works]
    df = pd.DataFrame(rows)
    for col in NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[NORMALIZED_COLUMNS]
