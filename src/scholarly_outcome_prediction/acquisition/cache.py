"""Deterministic cache for OpenAlex raw fetch. Keys by material request identity; reuse on exact match."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scholarly_outcome_prediction.logging_utils import get_logger
from scholarly_outcome_prediction.utils.io import load_json, load_jsonl, save_json, save_jsonl

logger = get_logger(__name__)

# Bump when fetch semantics or API usage change so old caches are not reused
OPENALEX_FETCH_CACHE_VERSION = "1"

CACHE_SUBDIR = "openalex_raw"
MANIFEST_FILENAME = "manifest.json"
DATA_FILENAME = "data.jsonl"


def build_fetch_identity(
    *,
    from_publication_date: str,
    to_publication_date: str,
    sample_size: int,
    seed: int,
    work_types: list[str] | None = None,
    sort: str | None = None,
    stratify_by_year: bool = False,
    use_random_sample: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable identity for the OpenAlex fetch request.
    Used only for cache keying; does not include output_path or dataset_name.
    """
    work_types_sorted = sorted(work_types) if work_types else []
    return {
        "from_publication_date": from_publication_date,
        "to_publication_date": to_publication_date,
        "sample_size": sample_size,
        "seed": seed,
        "work_types": work_types_sorted,
        "sort": sort,
        "stratify_by_year": stratify_by_year,
        "use_random_sample": use_random_sample,
        "_version": OPENALEX_FETCH_CACHE_VERSION,
    }


def compute_cache_key(identity: dict[str, Any]) -> str:
    """Deterministic short hash for the fetch identity. Same identity -> same key."""
    blob = json.dumps(identity, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def get_cache_dir(cache_root: Path, cache_key: str) -> Path:
    """Path to the cache entry directory for this key."""
    return cache_root / CACHE_SUBDIR / cache_key


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_FILENAME


def _data_path(cache_dir: Path) -> Path:
    return cache_dir / DATA_FILENAME


@dataclass
class FetchCacheLookup:
    """Result of cache lookup: hit with path to cached data, or miss."""

    hit: bool
    cache_key: str
    cache_dir: Path
    data_path: Path | None  # set when hit
    manifest: dict[str, Any] | None  # set when hit


def lookup(
    cache_root: Path,
    identity: dict[str, Any],
) -> FetchCacheLookup:
    """Check cache for an exact match. Returns FetchCacheLookup; hit=True only if valid entry exists."""
    cache_key = compute_cache_key(identity)
    cache_dir = get_cache_dir(cache_root, cache_key)
    data_path = _data_path(cache_dir)
    manifest_path = _manifest_path(cache_dir)

    if not cache_dir.exists() or not manifest_path.exists() or not data_path.exists():
        return FetchCacheLookup(hit=False, cache_key=cache_key, cache_dir=cache_dir, data_path=None, manifest=None)

    try:
        manifest = load_json(manifest_path)
    except Exception:
        logger.warning("Cache manifest unreadable at %s; treating as miss", manifest_path)
        return FetchCacheLookup(hit=False, cache_key=cache_key, cache_dir=cache_dir, data_path=None, manifest=None)

    if manifest.get("request_identity") != identity:
        logger.warning("Cache identity mismatch at %s; treating as miss", cache_dir)
        return FetchCacheLookup(hit=False, cache_key=cache_key, cache_dir=cache_dir, data_path=None, manifest=None)

    return FetchCacheLookup(hit=True, cache_key=cache_key, cache_dir=cache_dir, data_path=data_path, manifest=manifest)


def populate(
    cache_dir: Path,
    identity: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Write cache entry atomically: data.jsonl then manifest.json. Overwrites if exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = _data_path(cache_dir)
    manifest_path = _manifest_path(cache_dir)
    # Write to temp then rename for atomicity
    tmp_data = cache_dir / f".{DATA_FILENAME}.tmp"
    save_jsonl(records, tmp_data)
    tmp_data.rename(data_path)
    manifest = {
        "request_identity": identity,
        "row_count": len(records),
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "version": OPENALEX_FETCH_CACHE_VERSION,
    }
    save_json(manifest, manifest_path)
    logger.debug("Populated cache %s with %d rows", cache_dir, len(records))


@dataclass
class FetchResult:
    """Result of fetch_and_save: path written, and whether it came from cache."""

    output_path: Path
    from_cache: bool
    cache_key: str | None
    cache_path: str | None  # cache dir as string for provenance
    row_count: int


def copy_cached_to_output(cache_data_path: Path, output_path: Path) -> int:
    """Copy cached data.jsonl to output_path. Returns row count (from loading)."""
    records = load_jsonl(cache_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(records, output_path)
    return len(records)
