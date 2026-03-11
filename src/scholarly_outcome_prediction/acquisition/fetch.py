"""Fetch OpenAlex data and save raw JSONL. Supports deterministic cache by request identity."""

from __future__ import annotations

from pathlib import Path

from scholarly_outcome_prediction.acquisition.cache import (
    FetchResult,
    build_fetch_identity,
    compute_cache_key,
    copy_cached_to_output,
    get_cache_dir,
    lookup,
    populate,
)
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
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> FetchResult:
    """
    Fetch a reproducible sample of works from OpenAlex and save as JSONL.

    When stratify_by_year and use_random_sample are True (representative pilot), uses
    OpenAlex sample+seed per year for within-year random sampling (no API default order).
    When stratify_by_year is True and use_random_sample is False (temporal), uses cursor
    paging per year. When stratify_by_year is False, uses sort if provided.

    If cache_root is set and force_refresh is False, checks a deterministic cache keyed
    by the effective fetch request; on hit copies cached raw data to output_path and
    skips the API. On miss (or force_refresh), fetches from OpenAlex and populates cache.

    Returns FetchResult with output_path, from_cache, cache_key, cache_path, row_count.
    """
    output_path = Path(output_path)
    identity = build_fetch_identity(
        from_publication_date=from_publication_date,
        to_publication_date=to_publication_date,
        sample_size=sample_size,
        seed=seed,
        work_types=work_types,
        sort=sort,
        stratify_by_year=stratify_by_year,
        use_random_sample=use_random_sample,
    )

    if cache_root and not force_refresh:
        cache_root = Path(cache_root)
        result = lookup(cache_root, identity)
        if result.hit and result.data_path is not None:
            row_count = copy_cached_to_output(result.data_path, output_path)
            logger.info(
                "Reused OpenAlex raw cache (key=%s): copied %d rows to %s",
                result.cache_key,
                row_count,
                output_path,
            )
            return FetchResult(
                output_path=output_path,
                from_cache=True,
                cache_key=result.cache_key,
                cache_path=str(result.cache_dir),
                row_count=row_count,
            )

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
    save_jsonl(works, output_path)
    logger.info("Saved %d works to %s", len(works), output_path)

    if cache_root:
        cache_key = compute_cache_key(identity)
        cache_dir = get_cache_dir(Path(cache_root), cache_key)
        populate(cache_dir, identity, works)
        return FetchResult(
            output_path=output_path,
            from_cache=False,
            cache_key=cache_key,
            cache_path=str(cache_dir),
            row_count=len(works),
        )

    return FetchResult(
        output_path=output_path,
        from_cache=False,
        cache_key=None,
        cache_path=None,
        row_count=len(works),
    )
