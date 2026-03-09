"""Fetch OpenAlex data and save raw JSONL."""

from __future__ import annotations

from pathlib import Path

from scholarly_outcome_prediction.acquisition.openalex_client import (
    fetch_works_sample,
    fetch_works_sample_stratified,
    fetch_works_sample_stratified_representative,
)
from scholarly_outcome_prediction.logging_utils import get_logger
from scholarly_outcome_prediction.utils.io import save_jsonl
from scholarly_outcome_prediction.utils.seeds import set_global_seed

logger = get_logger(__name__)


def fetch_and_save(
    output_path: Path,
    sample_size: int,
    from_publication_date: str,
    to_publication_date: str,
    seed: int = 42,
    work_types: list[str] | None = None,
    sort: str | None = None,
    stratify_by_year: bool = False,
    use_random_sample: bool = False,
) -> Path:
    """
    Fetch a reproducible sample of works from OpenAlex and save as JSONL.

    When stratify_by_year and use_random_sample are True (representative pilot), uses
    OpenAlex sample+seed per year for within-year random sampling (no API default order).
    When stratify_by_year is True and use_random_sample is False (temporal), uses cursor
    paging per year. When stratify_by_year is False, uses sort if provided.
    Returns the path where raw data was written.
    """
    set_global_seed(seed)
    logger.info(
        "Fetching up to %d works from %s to %s (work_types=%s, sort=%s, stratify_by_year=%s, use_random_sample=%s)",
        sample_size,
        from_publication_date,
        to_publication_date,
        work_types,
        sort,
        stratify_by_year,
        use_random_sample,
    )
    if stratify_by_year and use_random_sample:
        works = fetch_works_sample_stratified_representative(
            from_date=from_publication_date,
            to_date=to_publication_date,
            sample_size=sample_size,
            seed=seed,
            work_types=work_types,
        )
    elif stratify_by_year:
        works = fetch_works_sample_stratified(
            from_date=from_publication_date,
            to_date=to_publication_date,
            sample_size=sample_size,
            seed=seed,
            work_types=work_types,
        )
    else:
        works = fetch_works_sample(
            from_date=from_publication_date,
            to_date=to_publication_date,
            sample_size=sample_size,
            seed=seed,
            work_types=work_types,
            sort=sort,
        )
    output_path = Path(output_path)
    save_jsonl(works, output_path)
    logger.info("Saved %d works to %s", len(works), output_path)
    return output_path
