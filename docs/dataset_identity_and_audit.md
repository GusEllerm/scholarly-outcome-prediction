# Dataset identity and overlap audit

## Intended difference between representative and temporal datasets

- **Representative** datasets are meant to approximate general learnability on a mixed sample across publication years. Sampling is **within-year random** (OpenAlex `sample` + `seed` per year) so the corpus is not biased by API default ordering. Use for random train/test splits and proxy or calendar-horizon targets when the goal is in-distribution performance.
- **Temporal** datasets are meant to support **forward-time generalization**: train on earlier years, test on later years. Sampling is **cursor-based per year** (no random sample): we take the first N works per year by API order. That yields a different population and ordering than the representative path, so benchmark comparisons between “representative” and “temporal” modes are methodologically distinct.

The two modes must **not** share the same fetch path. If both used random sampling with the same seed, they would produce the same (or nearly the same) set of works and only differ by output filename, which would invalidate comparisons.

## How dataset identity is recorded

- **Data config**
  - `dataset_mode`: optional `"representative"` or `"temporal"`. When set, it must match sampling:
    - `dataset_mode: temporal` → `use_random_sample: false` (cursor-based).
    - `dataset_mode: representative` with `stratify_by_year: true` → `use_random_sample: true` (within-year random).
  - Config validation enforces this so the two modes cannot silently collapse to the same effective identity.
- **Validation/provenance**
  - Dataset validation reports (e.g. `artifacts/reports/{dataset_id}_dataset_validation.json`) include a **provenance** block:
    - `dataset_id`, `processed_path`, `dataset_mode`
    - `source_config_path`, `generation_params` (stratify_by_year, use_random_sample, seed, sample_size, dates, work_types)
    - `selection_strategy_summary` (human-readable description of how the sample was built)
    - `work_id_fingerprint`: short hash of sorted `openalex_id` values so two datasets can be compared even when other stats look similar.

## How to run the overlap audit

After generating both representative and temporal processed datasets (fetch + prepare for each config):

```bash
# CLI
uv run scholarly-outcome-prediction audit-dataset-overlap \
  --left data/processed/openalex_representative_articles_1000.parquet \
  --right data/processed/openalex_temporal_articles_1000.parquet \
  --label-left representative \
  --label-right temporal \
  --out-dir artifacts/diagnostics

# Or use the Make target
make audit-dataset-overlap
```

Outputs (in `artifacts/diagnostics/` or the given `--out-dir`):

- **JSON**: sizes, overlap count, overlap rate from both directions, `identical` flag, counts of IDs only in left/right, and sample IDs (overlap, only-left, only-right).
- **Markdown**: human-readable summary and the same sample IDs.

Interpretation:

- **Identical** = same set of work IDs in both files (overlap = size of each). With the hardened configs (temporal using cursor sampling), this should not happen after a fresh fetch for both.
- **Partial overlap** = some works appear in both; the rest are unique to each. Expected when both span the same year range but use different sampling.
- **Zero overlap** = disjoint ID sets. Possible but not required; what matters is that overlap is **measured and explainable**, not forced to zero.

Re-run the audit after any change to data configs or after regenerating datasets to confirm distinctness.

## Why this matters for benchmarking and caching

- **Benchmarking**: Representative vs temporal metrics are compared in the same report. If the underlying datasets were the same, those comparisons would be misleading. Hardening ensures they are built differently and that provenance is explicit.
- **Caching**: Cache keys for fetch/prepare should include full generation parameters (or content identity), not only dataset labels. The provenance and overlap audit support designing cache keys that do not confuse representative and temporal data. See `docs/diagnostics/caching_readiness_review.md`.
