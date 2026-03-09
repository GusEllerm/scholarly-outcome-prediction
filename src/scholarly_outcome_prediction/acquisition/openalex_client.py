"""Minimal OpenAlex API client using requests.

Fetch strategies:
- Default (sort=None): Cursor pagination, then sort by id and take first N. API default order
  can be relevance/citation-biased; with sort=publication_date:asc you get oldest-first → single-year slice.
- Stratified by year (stratify_by_year=True): Fetch per year from the configured range, then combine.
  Ensures the sample spans multiple years for representative or temporal benchmarks.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import requests

from scholarly_outcome_prediction.logging_utils import get_logger

logger = get_logger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _years_from_date_range(from_date: str, to_date: str) -> list[int]:
    """Parse YYYY-MM-DD range into list of years (inclusive)."""
    try:
        start_year = int(from_date[:4])
        end_year = int(to_date[:4])
    except (ValueError, TypeError):
        return []
    return list(range(start_year, end_year + 1))


def _get_mailto() -> str | None:
    return os.environ.get("OPENALEX_MAILTO", "").strip() or None


def _build_headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "scholarly-outcome-prediction/0.1"}
    mailto = _get_mailto()
    if mailto:
        headers["Mailto"] = mailto
    return headers


def fetch_works_page(
    from_date: str,
    to_date: str,
    per_page: int = 25,
    cursor: str | None = None,
    sample: int | None = None,
    seed: int | None = None,
    work_types: list[str] | None = None,
    sort: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Fetch one page of works from OpenAlex.

    Returns (list of work dicts, next_cursor or None).
    Filter: from_publication_date, to_publication_date; optionally type (work_types).
    When sample is set, use seed for reproducible random sample (OpenAlex API); use basic paging, not cursor.
    """
    filter_parts = [f"from_publication_date:{from_date}", f"to_publication_date:{to_date}"]
    if work_types:
        filter_parts.append("type:" + "|".join(work_types))
    params: dict[str, Any] = {
        "filter": ",".join(filter_parts),
        "per-page": min(per_page, 200),
    }
    if cursor and (sample is None or sample <= 0):
        params["cursor"] = cursor
    if sample is not None and sample > 0:
        params["sample"] = min(sample, 10000)  # OpenAlex cap
        if seed is not None:
            params["seed"] = seed
    if sort:
        params["sort"] = sort

    resp = requests.get(
        OPENALEX_WORKS_URL,
        params=params,
        headers=_build_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    meta = data.get("meta") or {}
    next_cursor = meta.get("next_cursor")
    return results, next_cursor


def fetch_works_sample(
    from_date: str,
    to_date: str,
    sample_size: int,
    seed: int = 42,
    work_types: list[str] | None = None,
    sort: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch a reproducible sample of works: paginate with date filter (and optional type/sort), sort by id, take first N.

    Does not use API search or seed. Optional sort (e.g. publication_date:asc) reduces bias from API default order.
    """
    all_works: list[dict[str, Any]] = []
    cursor: str | None = "*"
    per_page = 200

    while len(all_works) < sample_size:
        results, next_cursor = fetch_works_page(
            from_date=from_date,
            to_date=to_date,
            per_page=per_page,
            cursor=cursor,
            sample=None,
            work_types=work_types,
            sort=sort,
        )
        if not results:
            break
        all_works.extend(results)
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.2)

    # Reproducible: sort by id and take first sample_size
    def _id(w: dict[str, Any]) -> str:
        return str(w.get("id") or "")

    all_works.sort(key=_id)
    out = all_works[:sample_size]
    if len(out) < sample_size:
        logger.warning("Requested %d works, got %d", sample_size, len(out))
    return out


def fetch_works_sample_stratified(
    from_date: str,
    to_date: str,
    sample_size: int,
    seed: int = 42,
    work_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch a reproducible sample that spans the full year range (representative/temporal).

    Fetches per year from the date range (no sort=publication_date:asc), then combines,
    dedupes by id, sorts by id, and takes first sample_size. Avoids oldest-first collapse.
    Does not use API search, sample, or seed; uses cursor paging per year.
    """
    years = _years_from_date_range(from_date, to_date)
    if not years:
        logger.warning("Could not parse year range from %s to %s", from_date, to_date)
        return []
    per_year = max(1, math.ceil(sample_size / len(years)))
    all_works: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for year in years:
        y_from = f"{year}-01-01"
        y_to = f"{year}-12-31"
        cursor: str | None = "*"
        per_page = 200
        year_works: list[dict[str, Any]] = []
        while len(year_works) < per_year:
            results, next_cursor = fetch_works_page(
                from_date=y_from,
                to_date=y_to,
                per_page=per_page,
                cursor=cursor,
                sample=None,
                work_types=work_types,
                sort=None,
            )
            if not results:
                break
            for w in results:
                wid = str(w.get("id") or "")
                if wid and wid not in seen_ids:
                    seen_ids.add(wid)
                    year_works.append(w)
                    if len(year_works) >= per_year:
                        break
            if len(year_works) >= per_year or not next_cursor:
                break
            cursor = next_cursor
            time.sleep(0.2)
        all_works.extend(year_works)
        logger.debug("Year %d: got %d works", year, len(year_works))

    def _id(w: dict[str, Any]) -> str:
        return str(w.get("id") or "")

    all_works.sort(key=_id)
    out = all_works[:sample_size]
    if len(out) < sample_size:
        logger.warning("Requested %d works (stratified), got %d", sample_size, len(out))
    return out


def fetch_works_random_sample(
    from_date: str,
    to_date: str,
    sample_size: int,
    seed: int,
    work_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch a random sample from OpenAlex using API sample+seed (one request).
    Returns up to sample_size works; reproducible for the same filter+seed.
    """
    params_filter = [f"from_publication_date:{from_date}", f"to_publication_date:{to_date}"]
    if work_types:
        params_filter.append("type:" + "|".join(work_types))
    n = min(sample_size, 10000)
    results, _ = fetch_works_page(
        from_date=from_date,
        to_date=to_date,
        per_page=n,
        cursor=None,
        sample=n,
        seed=seed,
        work_types=work_types,
        sort=None,
    )
    return results or []


def fetch_works_sample_stratified_representative(
    from_date: str,
    to_date: str,
    sample_size: int,
    seed: int = 42,
    work_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Representative pilot: within-year RANDOM sampling via OpenAlex sample+seed per year.

    Does not use API default ordering. For each year in the range, requests a random
    sample (sample + seed) so the combined corpus is balanced across years and not
    biased toward highly cited works. Reproducible via seed; seed per year is
    derived as seed * 1000 + year so years get different but deterministic samples.
    """
    years = _years_from_date_range(from_date, to_date)
    if not years:
        logger.warning("Could not parse year range from %s to %s", from_date, to_date)
        return []
    per_year = max(1, math.ceil(sample_size / len(years)))
    per_year = min(per_year, 10000)
    logger.info(
        "Representative sampling: random sample per year (sample=%s, seed=base %s), years=%s",
        per_year,
        seed,
        years,
    )
    all_works: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for year in years:
        y_from = f"{year}-01-01"
        y_to = f"{year}-12-31"
        year_seed = seed * 1000 + year
        results = fetch_works_random_sample(
            from_date=y_from,
            to_date=y_to,
            sample_size=per_year,
            seed=year_seed,
            work_types=work_types,
        )
        for w in results:
            wid = str(w.get("id") or "")
            if wid and wid not in seen_ids:
                seen_ids.add(wid)
                all_works.append(w)
        logger.info("Year %d: requested %d, got %d (random sample, seed=%s)", year, per_year, len(results), year_seed)
        time.sleep(0.2)

    def _id(w: dict[str, Any]) -> str:
        return str(w.get("id") or "")

    all_works.sort(key=_id)
    out = all_works[:sample_size]
    if len(out) < sample_size:
        logger.warning("Requested %d works (representative stratified random), got %d", sample_size, len(out))
    return out
