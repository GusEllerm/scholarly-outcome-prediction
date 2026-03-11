"""Tests for OpenAlex raw fetch cache: key identity, hit/miss, refresh, provenance."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scholarly_outcome_prediction.acquisition.cache import (
    OPENALEX_FETCH_CACHE_VERSION,
    build_fetch_identity,
    compute_cache_key,
    copy_cached_to_output,
    get_cache_dir,
    lookup,
    populate,
)
from scholarly_outcome_prediction.acquisition.fetch import fetch_and_save
from scholarly_outcome_prediction.utils.io import load_jsonl, save_jsonl


def test_same_identity_same_cache_key() -> None:
    """Identical effective fetch request -> same cache key."""
    identity = build_fetch_identity(
        from_publication_date="2018-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=42,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=True,
    )
    key1 = compute_cache_key(identity)
    key2 = compute_cache_key(identity)
    assert key1 == key2
    assert len(key1) == 16


def test_different_identity_different_cache_key() -> None:
    """Materially different request -> different cache key."""
    base = build_fetch_identity(
        from_publication_date="2018-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=42,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=True,
    )
    key_base = compute_cache_key(base)

    diff_seed = {**base, "seed": 99}
    assert compute_cache_key(diff_seed) != key_base

    diff_sample = {**base, "sample_size": 500}
    assert compute_cache_key(diff_sample) != key_base

    diff_dates = {**base, "from_publication_date": "2019-01-01"}
    assert compute_cache_key(diff_dates) != key_base

    diff_stratify = {**base, "stratify_by_year": False}
    assert compute_cache_key(diff_stratify) != key_base

    diff_random = {**base, "use_random_sample": False}
    assert compute_cache_key(diff_random) != key_base


def test_work_types_order_normalized() -> None:
    """work_types list order should not affect key (sorted in identity)."""
    a = build_fetch_identity(
        from_publication_date="2018-01-01",
        to_publication_date="2018-12-31",
        sample_size=100,
        seed=1,
        work_types=["article", "book"],
    )
    b = build_fetch_identity(
        from_publication_date="2018-01-01",
        to_publication_date="2018-12-31",
        sample_size=100,
        seed=1,
        work_types=["book", "article"],
    )
    assert compute_cache_key(a) == compute_cache_key(b)


def test_representative_and_temporal_same_fetch_identity_same_key() -> None:
    """When representative and temporal use same date range, size, seed, stratify, use_random_sample they share fetch identity."""
    rep = build_fetch_identity(
        from_publication_date="2015-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=99,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=True,
    )
    temp = build_fetch_identity(
        from_publication_date="2015-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=99,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=True,
    )
    assert compute_cache_key(rep) == compute_cache_key(temp)


def test_representative_and_temporal_different_fetch_different_key() -> None:
    """Different sampling (e.g. temporal cursor vs representative random) -> different key."""
    rep = build_fetch_identity(
        from_publication_date="2015-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=42,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=True,
    )
    temp = build_fetch_identity(
        from_publication_date="2015-01-01",
        to_publication_date="2020-12-31",
        sample_size=1000,
        seed=99,
        work_types=["article"],
        stratify_by_year=True,
        use_random_sample=False,
    )
    assert compute_cache_key(rep) != compute_cache_key(temp)


def test_cache_lookup_miss_empty_dir() -> None:
    """Lookup with no cache dir -> miss."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        identity = build_fetch_identity(
            from_publication_date="2018-01-01",
            to_publication_date="2018-12-31",
            sample_size=10,
            seed=1,
        )
        result = lookup(root, identity)
        assert result.hit is False
        assert result.cache_key is not None
        assert result.data_path is None


def test_cache_populate_and_lookup_hit() -> None:
    """Populate cache then lookup -> hit and data path present."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        identity = build_fetch_identity(
            from_publication_date="2018-01-01",
            to_publication_date="2018-12-31",
            sample_size=10,
            seed=1,
        )
        records = [{"id": f"W{i}", "type": "article"} for i in range(5)]
        cache_dir = get_cache_dir(root, compute_cache_key(identity))
        populate(cache_dir, identity, records)
        result = lookup(root, identity)
        assert result.hit is True
        assert result.data_path is not None
        assert result.data_path.exists()
        loaded = load_jsonl(result.data_path)
        assert len(loaded) == 5


def test_copy_cached_to_output() -> None:
    """Copy cached data to output path; row count returned."""
    with tempfile.TemporaryDirectory() as d:
        cache_file = Path(d) / "data.jsonl"
        save_jsonl([{"id": "1"}, {"id": "2"}], cache_file)
        out = Path(d) / "out" / "raw.jsonl"
        n = copy_cached_to_output(cache_file, out)
        assert n == 2
        assert out.exists()
        assert len(load_jsonl(out)) == 2


def test_fetch_uses_cache_on_hit_and_skips_api() -> None:
    """When cache has entry, fetch_and_save copies from cache and does not call OpenAlex."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cache_root = root / "cache"
        out_path = root / "data" / "raw" / "test.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        identity = build_fetch_identity(
            from_publication_date="2018-01-01",
            to_publication_date="2018-12-31",
            sample_size=5,
            seed=42,
            work_types=["article"],
            stratify_by_year=True,
            use_random_sample=True,
        )
        from scholarly_outcome_prediction.acquisition.cache import get_cache_dir, compute_cache_key

        cache_dir = get_cache_dir(cache_root, compute_cache_key(identity))
        cache_dir.mkdir(parents=True, exist_ok=True)
        from scholarly_outcome_prediction.acquisition.cache import DATA_FILENAME, MANIFEST_FILENAME
        from scholarly_outcome_prediction.utils.io import save_json

        save_jsonl([{"id": f"W{i}"} for i in range(5)], cache_dir / DATA_FILENAME)
        save_json(
            {"request_identity": identity, "row_count": 5, "version": OPENALEX_FETCH_CACHE_VERSION},
            cache_dir / MANIFEST_FILENAME,
        )
        with patch("scholarly_outcome_prediction.acquisition.fetch.fetch_works_sample_stratified_representative") as m:
            result = fetch_and_save(
                output_path=out_path,
                sample_size=5,
                from_publication_date="2018-01-01",
                to_publication_date="2018-12-31",
                seed=42,
                work_types=["article"],
                stratify_by_year=True,
                use_random_sample=True,
                force_refresh=False,
                cache_root=cache_root,
            )
            m.assert_not_called()
        assert result.from_cache is True
        assert result.cache_key is not None
        assert result.row_count == 5
        assert out_path.exists()
        assert len(load_jsonl(out_path)) == 5


def test_fetch_force_refresh_bypasses_cache() -> None:
    """With force_refresh=True, API is called even if cache exists."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cache_root = root / "cache"
        out_path = root / "out.jsonl"
        with patch("scholarly_outcome_prediction.acquisition.fetch.fetch_works_sample_stratified_representative") as m:
            m.return_value = [{"id": "X1"}, {"id": "X2"}]
            result = fetch_and_save(
                output_path=out_path,
                sample_size=2,
                from_publication_date="2018-01-01",
                to_publication_date="2018-12-31",
                seed=1,
                stratify_by_year=True,
                use_random_sample=True,
                force_refresh=True,
                cache_root=cache_root,
            )
            m.assert_called_once()
        assert result.from_cache is False
        assert result.row_count == 2


def test_provenance_records_cache_usage() -> None:
    """Validation run_validation_and_save accepts and stores raw_fetch_from_cache, openalex_cache_key."""
    import tempfile

    import pandas as pd

    from scholarly_outcome_prediction.validation.dataset_validation import run_validation_and_save

    df = pd.DataFrame({
        "publication_year": [2018] * 50 + [2019] * 50,
        "type": ["article"] * 100,
        "cited_by_count": [1] * 100,
        "venue_name": ["J"] * 100,
        "openalex_id": [f"W{i}" for i in range(100)],
    })
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        processed = Path(d) / "test.parquet"
        df.to_parquet(processed, index=False)
        result, _, _ = run_validation_and_save(
            raw_records=None,
            df=df,
            processed_path=processed,
            out_dir=out_dir,
            dataset_mode="temporal",
            raw_fetch_from_cache=True,
            openalex_cache_key="abc123",
            openalex_cache_path="/cache/openalex_raw/abc123",
        )
        prov = result.get("provenance", {})
        assert prov.get("raw_fetch_from_cache") is True
        assert prov.get("openalex_cache_key") == "abc123"
        assert prov.get("openalex_cache_path") == "/cache/openalex_raw/abc123"
