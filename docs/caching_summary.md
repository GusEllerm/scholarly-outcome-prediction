# Caching implementation summary

Short summary of the deterministic, stage-aware caching added for OpenAlex-backed dataset generation. Full details: [dataset_caching.md](dataset_caching.md).

---

## What is cached now

- **OpenAlex raw fetch** only.  
  Location: `artifacts/cache/openalex_raw/<key>/` (data.jsonl + manifest.json).  
  Repeated runs with the same effective request reuse this cache and do not call the API.

---

## What defines cache identity

- Deterministic hash of: **from_publication_date**, **to_publication_date**, **sample_size**, **seed**, **work_types** (sorted), **sort**, **stratify_by_year**, **use_random_sample**, and a **version** string.  
- Not included: output path, dataset name, or benchmark mode. So representative and temporal configs share the same raw cache when their fetch parameters are identical.

---

## What is not yet cached

- Normalize (raw → processed parquet).
- Target construction, feature matrices, splits.
- Model training or evaluation.

---

## How users force a refresh

- **fetch:** `uv run scholarly-outcome-prediction fetch --config <data-config> --force-refresh`
- **run:** `uv run scholarly-outcome-prediction run --data-config ... --baseline-config ... --xgb-config ... --force-refresh`

Without `--force-refresh`, cache is used on exact identity match. To clear all cache: `rm -rf artifacts/cache`.
