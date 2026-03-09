# Prompt 4 Debug Report: Query Audit, Dataset Identity, Venue, Time Split

## 1. OpenAlex request construction

### 1.1 Old pilot config (`configs/data/openalex_pilot.yaml`)

The pipeline builds the OpenAlex request as follows:

- **Endpoint:** `GET https://api.openalex.org/works`
- **Query parameters (per page):**
  - `filter`: `from_publication_date:2015-01-01,to_publication_date:2020-12-31`
  - `per-page`: `200`
  - `cursor`: `*` on first page, then `meta.next_cursor` for subsequent pages
- **Not used:** `search`, `sort`, `sample`, `seed`, `select`, `type` (work type filter)

So the **old pilot request used no search term, no sort, no sample/seed, and no work-type filter**. The API returns results in its default order (typically relevance/citation-weighted). Pagination then takes the first 1000 works from that default order, which is why the pilot corpus was biased toward highly cited works.

### 1.2 New pilot config (`configs/data/openalex_pilot_articles.yaml`)

- **Same endpoint and paging.**
- **Query parameters:**
  - `filter`: `from_publication_date:2015-01-01,to_publication_date:2020-12-31,type:article`
  - `per-page`: `200`
  - `cursor`: as above
  - `sort`: `publication_date:asc`
- **Still not used:** `search`, `sample`, `seed`, `select`

So the **new pilot request** restricts to OpenAlex type `article` (journal/proceedings articles) and sorts by `publication_date:asc`, giving a chronological, less citation-biased sample. Note: OpenAlex uses `type:article`, not `journal-article` or `proceedings-article` (those return 0 results).

### 1.3 Query audit summary

| Parameter | Old pilot | New pilot (articles) |
|-----------|-----------|------------------------|
| search    | No        | No                     |
| sort      | No (API default) | Yes: `publication_date:asc` |
| sample    | No        | No                     |
| seed      | No        | No                     |
| filter    | dates only | dates + `type:journal-article\|proceedings-article` |
| select    | No        | No                     |
| work types constrained | No | Yes                    |

---

## 2. Work type and bias diagnosis

- **Observation:** The old pilot included mixed work types (e.g. book-chapter, book, reference-entry). Some of these (e.g. book-chapter from ebook platforms with empty authorships) had very high `cited_by_count`, skewing the corpus.
- **Root cause:** No `type` filter and no `sort` meant the API’s default ordering (often citation-influenced) dominated, and all work types were included.
- **Recommendation:** For a defensible pilot benchmark, restrict to OpenAlex type **article** (journal/proceedings articles) and use **sort=publication_date:asc**. This is implemented in the new config `openalex_pilot_articles.yaml`; the old `openalex_pilot.yaml` is unchanged.

---

## 3. Dataset identity repair

- **Problem:** Run artifacts (e.g. `baseline_regression.json`, `xgb_regression.json`) could record `dataset_id` (and related fields) from the **experiment config** instead of the **actual dataset used** in the run, so artifacts could disagree or be stale.
- **Fix:**
  - **Run path (`run_pipeline_from_configs`):** The **actual** dataset is the one produced by the data pipeline: `data_cfg.dataset_name` and `processed_path = root / "data" / "processed" / f"{data_cfg.dataset_name}.parquet"`. All metrics artifacts now receive:
    - `effective_dataset_id` = `data_cfg.dataset_name`
    - `effective_processed_path` = `str(processed_path)`
    - `data_config_path` = path to the data config used
    - `experiment_config_path` = path to the experiment config used
  - **Standalone `evaluate`:** Uses the file at `cfg.data.processed_path`; artifacts now record `effective_dataset_id` = stem of that path and `effective_processed_path` = resolved path.
- **Schema:** `build_run_metadata` now accepts and persists `effective_dataset_id`, `effective_processed_path`, `data_config_path`, and `experiment_config_path`. The persisted `dataset_id` in the JSON is set from `effective_dataset_id` when provided, so a single run cannot claim a different dataset than the one actually used.

---

## 4. Venue name diagnosis and fix

- **Problem:** `venue_name` was 100% missing in the processed pilot because the normalizer only looked at `primary_location.display_name` and `primary_location.venue.display_name`. In OpenAlex, the journal/serial name is often under **`primary_location.source.display_name`**; when `source` is null (e.g. book-chapter), **`primary_location.raw_source_name`** is present.
- **Fix:** `_venue_name` in `schemas.py` now uses a clear fallback chain:
  1. `primary_location.source.display_name`
  2. `primary_location.raw_source_name`
  3. `primary_location.display_name`
  4. `primary_location.venue.display_name`
  5. `host_venue.display_name` and `host_venue.source.display_name`
  6. `locations[0].source.display_name` (when primary is missing or uninformative)
- **Work types:** For normal journal/proceedings articles, `source.display_name` or `raw_source_name` typically populates `venue_name`. Types without a meaningful venue (e.g. some paratext) may still have null; that is documented and left as-is.

---

## 5. Time-based split

- **Implementation:** In `data/split.py`, when `split_kind == "time"` and `time_column` is set (e.g. `publication_year`):
  - Rows with null `time_column` are dropped for the split.
  - The dataframe is sorted by `time_column` (ascending).
  - The last `floor(len * test_size)` rows form the **test** set; the rest form the **train** set.
- **Config:** Experiment config uses `split_kind: time` and `time_column: publication_year` (e.g. `configs/experiments/baseline_regression_time.yaml`).
- **Determinism:** No randomness; same data and config always yield the same split.
- **Validation:** If `time_column` is missing or not in the dataframe, or all values are null, the split raises a clear `ValueError`.
- **Random split:** `split_kind: random` is unchanged for debugging and comparison.

---

## 6. Limitations

- **Proxy target:** The benchmark still uses the proxy target `cited_by_count`; fixed-horizon outcome is not implemented.
- **No new models:** Only existing model families are used; no new model types were added.
- **Venue nulls:** Some work types (e.g. certain book or paratext records) may legitimately have no venue; they remain null.
- **OpenAlex only:** Fetch and query logic are specific to OpenAlex; no other data sources are wired.
- **Time split:** Uses a single chronological cutoff (last N% by time); no multiple cutoffs or expanding window.

---

## 7. Files changed (summary)

- **Acquisition:** `openalex_client.py` (optional `work_types`, `sort`); `fetch.py` (pass-through); `settings.py` (`DataConfig.work_types`, `sort`).
- **Data:** `schemas.py` (`_venue_name` fallback chain); `split.py` (time-based split implementation).
- **Evaluation:** `report.py` (`build_run_metadata` extended with effective dataset and config paths).
- **CLI:** `cli.py` (fetch passes work_types/sort; run path passes effective dataset and config paths; evaluate passes effective dataset/path).
- **Configs:** New `configs/data/openalex_pilot_articles.yaml`; new `configs/experiments/baseline_regression_time.yaml`.
- **Docs/artifacts:** This report and `artifacts/debug/*.json` (see below).

---

## 8. Machine-readable audits

See:

- `artifacts/debug/openalex_query_audit.json`
- `artifacts/debug/dataset_identity_audit.json`
- `artifacts/debug/venue_extraction_audit.json`
- `artifacts/debug/time_split_audit.json`
