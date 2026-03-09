# Benchmark modes: representative vs temporal

The pipeline supports two explicit pilot benchmark modes. They use different **data configs**, **split behaviour**, and **Makefile targets**.

## Why two modes?

- **Representative sampling** aims for a fair, broad sample across the configured year range so that model comparison and feature sanity checks are not biased by a single year or an “oldest-first” slice.
- **Temporal evaluation** aims to test generalization over time: train on past publications, evaluate on future ones. That requires real year spread in the data and a time-based split, not a random split.

Using one fetch/split setup for both goals would be ambiguous (e.g. random split on oldest-first data is not representative; time split on single-year data is invalid). So the repo separates them into **representative pilot** and **temporal pilot** with distinct configs and commands.

## Representative pilot

**Purpose:** Broad laptop-scale benchmarking, sanity checks on feature usefulness, fairer corpus composition.

**Data:**
- Config: `configs/data/openalex_representative_articles_1000.yaml`
- Article-only (`work_types: [article]`), date range 2015–2020
- **Within-year random sampling:** `stratify_by_year: true` and `use_random_sample: true` so the fetch uses OpenAlex **sample + seed** per year. This gives a true random sample within each year instead of the API default order (which is biased toward highly cited works). Reproducible via seed; each year uses `seed * 1000 + year` so the corpus is not elite-only.

**Split:** Random (e.g. 80% train / 20% test) for standard benchmarking.

**Experiments:** `baseline_representative`, `xgb_representative` (see `configs/experiments/`).

**Run:**
```bash
make run-representative-pilot
```

## Temporal pilot

**Purpose:** Temporal generalization — train on earlier years, test on later years.

**Data:**
- Config: `configs/data/openalex_temporal_articles_1000.yaml`
- Same article-only, 2015–2020, **stratify_by_year: true** so the dataset has real year spread

**Split:** Time-based with **explicit year boundaries** in experiment config:
- `split_kind: time`
- `time_column: publication_year`
- `train_year_end: 2018` → train set = rows with `publication_year <= 2018`
- `test_year_start: 2019` → test set = rows with `publication_year >= 2019`

The split fails clearly if the dataset has only one distinct year (no temporal variation).

**Experiments:** `baseline_temporal`, `xgb_temporal`.

**Run:**
```bash
make run-temporal-pilot
```

## Validation

After prepare (or after a full run), the pipeline runs **dataset validation** and writes reports to `artifacts/reports/<run_id>_dataset_validation.json` and `.md`. Validation checks:

- Row count above a minimum
- Publication year spread (at least 2 distinct years when multi-year is expected)
- Venue name non-null rate (fails if missingness exceeds a threshold, e.g. 95%)
- **Representative realism:** For `dataset_mode: representative`, validation also fails if median citations exceed a threshold (default 500) or distinct venue count is below a minimum (default 10), so obviously elite-only or narrow corpora are caught before training.

If validation fails, the pipeline exits with an error. You can also run validation alone:

```bash
make validate-latest-pilot
# or
scholarly-outcome-prediction validate --data-config configs/data/openalex_representative_articles_1000.yaml
```

## Proxy target

Both modes currently use the **proxy** target (`cited_by_count` with optional `log1p` transform). A future **research** target (e.g. fixed-horizon citations) is not implemented yet; config uses `target_mode: proxy` and run artifacts record it.
