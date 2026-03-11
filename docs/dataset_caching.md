# Dataset caching (OpenAlex raw fetch)

The pipeline caches **raw OpenAlex fetch** results by a deterministic request identity. Repeated runs with the same effective data config reuse the cache and do not re-hit the API.

---

## What is cached

| Stage | Cached | Location |
|-------|--------|----------|
| **OpenAlex raw fetch** | Yes | `artifacts/cache/openalex_raw/<key>/` |
| Normalize (raw → processed) | No | — |
| Model training / features / splits | No | — |

Each cache entry is a directory containing:

- `data.jsonl` — raw work records (same as would be written to `data/raw/...`)
- `manifest.json` — request identity, row count, created time, version

---

## What determines cache identity

The cache key is a **hash of the effective fetch request**, not of dataset name or benchmark mode.

Included in the identity:

- `from_publication_date`, `to_publication_date`
- `sample_size`, `seed`
- `work_types` (sorted)
- `sort`
- `stratify_by_year`, `use_random_sample`
- A fixed **version** string (bumped when fetch semantics change)

Not included: `output_path`, `dataset_name`, `dataset_mode`. So two data configs (e.g. representative and temporal) that differ only by name or mode but use the same date range, size, seed, and sampling settings **share the same raw cache**.

If you change any of the above (e.g. seed, sample_size, or use_random_sample), a new cache key is used and the API is called on the next run.

---

## How to refresh or bypass the cache

- **`fetch` command:**  
  `uv run scholarly-outcome-prediction fetch --config configs/data/foo.yaml --force-refresh`  
  Re-downloads from OpenAlex and overwrites the cache entry for that request.

- **`run` command:**  
  `uv run scholarly-outcome-prediction run --data-config ... --baseline-config ... --xgb-config ... --force-refresh`  
  Same: bypasses cache for the data config’s fetch and re-populates cache.

Without `--force-refresh`, cache is used when the computed key matches an existing entry.

---

## Provenance: how to see cache usage

Dataset validation reports (e.g. `artifacts/reports/<dataset_id>_dataset_validation.json`) include in **provenance**:

- **`raw_fetch_from_cache`** — `true` if raw data was copied from cache; `false` if fetched from the API.
- **`openalex_cache_key`** — cache key (16-char hash) when cache was used or populated.
- **`openalex_cache_path`** — path to the cache entry directory when cache was used or populated.

The markdown report lists “Raw fetch from cache” and “OpenAlex cache key” so you can tell at a glance whether a run reused cache.

---

## Current limitations

- Only the **raw fetch** stage is cached. Normalization (raw → processed parquet) and later stages are not cached; they run every time.
- Cache is **exact match only**. There is no “best effort” or fuzzy matching.
- Cache directory is **local** (`artifacts/cache/`). It is not shared across machines or CI runs unless you copy or mount it.
- `artifacts/cache/` is in `.gitignore`; cache is not committed.

---

## Clearing the cache

Remove the cache directory to force all future fetches to hit the API:

```bash
rm -rf artifacts/cache
```

Or remove a single key’s directory under `artifacts/cache/openalex_raw/<key>/` to invalidate only that request.
