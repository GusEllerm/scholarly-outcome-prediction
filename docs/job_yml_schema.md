# Config YAML schema reference

This document describes the two main YAML config types used by the pipeline: **data configs** (fetch/prepare) and **experiment configs** (train/evaluate). All paths in configs are relative to the project root unless otherwise noted.

---

## 1. Data config (`configs/data/*.yaml`)

Used by: `fetch`, `prepare`, `run` (for the fetch+prepare step), `validate` (with `--data-config`).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **dataset_name** | string | Yes | — | Short identifier for the dataset (e.g. `openalex_temporal_articles_1000`). Used to derive the processed parquet path as `data/processed/{dataset_name}.parquet` when using `run`. |
| **sample_size** | integer | Yes | — | Number of works to fetch (must be > 0 and ≤ 100000). |
| **seed** | integer | No | 42 | Random seed for reproducible sampling. |
| **from_publication_date** | string | No | `"2018-01-01"` | Start of publication date range (ISO date). |
| **to_publication_date** | string | No | `"2018-12-31"` | End of publication date range (ISO date). |
| **output_path** | string | No | `"data/raw/openalex_sample.jsonl"` | Where to write raw API responses (JSONL). |
| **fields** | list of strings | No | `[]` | API fields to request (optional). |
| **work_types** | list of strings or null | No | null | Restrict work types (e.g. `["article"]` for OpenAlex). |
| **sort** | string or null | No | null | API sort order. Do **not** use for representative sampling (causes oldest-first slice). |
| **stratify_by_year** | boolean | No | false | If true, fetch per-year and combine so the sample spans the full date range. |
| **use_random_sample** | boolean | No | false | If true **and** stratify_by_year is true, use OpenAlex sample+seed per year for within-year random sampling (representative). If false with stratify_by_year, use cursor paging (temporal; order not randomized within year). |

---

## 2. Experiment config (`configs/experiments/*.yaml`)

Used by: `train`, `evaluate`, `run` (for the train+evaluate step). Top-level and nested blocks.

### 2.1 Top-level fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **experiment_name** | string | No | `"experiment"` | Unique name for this run. Determines artifact filenames: `artifacts/models/{experiment_name}.joblib`, `artifacts/metrics/{experiment_name}.json`. |
| **task_type** | string | No | `"regression"` | Task type. Currently only `regression` is fully supported. |
| **data** | object | No | see below | Paths and dataset id for this experiment. |
| **target** | object | No | see below | Target variable, transform, and semantics. |
| **features** | object | No | see below | Numeric and categorical feature lists. |
| **split** | object | No | see below | Train/test split settings. |
| **model** | object | No | see below | Model name and hyperparameters. |
| **evaluation** | object | No | see below | Metrics to compute. |
| **benchmark** | object | **Yes** (current jobs) | — | **Required for train/evaluate/run.** Explicit benchmark metadata for run artifacts and comparison. When omitted, `train` / `evaluate` / `run` fail with a clear error. See §2.8. |
| **ablation** | object or null | No | null | When present, this run is an ablation; `name` and `features_removed` (non-empty list) are required. See §2.9. |

---

### 2.2 `data` block (DataPathsConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **processed_path** | string | No | `"data/processed/openalex_sample_100.parquet"` | Path to the processed parquet file (relative to project root). |
| **dataset_id** | string or null | No | null | Dataset identifier for run metadata (e.g. same as data config’s `dataset_name`). |

---

### 2.3 `target` block (TargetConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **name** | string | No | `"cited_by_count"` | Target column name in the processed dataframe. |
| **transform** | string or null | No | `"log1p"` | Transform applied to target: `"log1p"` or null. |
| **target_mode** | string | No | `"proxy"` | One of: `proxy`, `research`, `calendar_horizon`. `proxy` = bootstrap proxy (e.g. present-day cited_by_count). `calendar_horizon` = citations summed over a fixed calendar-year window from `counts_by_year`. `research` = future fixed-horizon (not fully implemented). |
| **source** | string or null | **Yes** when `target_mode: calendar_horizon` | null | For `calendar_horizon`: source field (e.g. `"counts_by_year"`). Required when target_mode is `calendar_horizon`. |
| **horizon_years** | integer or null | **Yes** when `target_mode: calendar_horizon` | null | For `calendar_horizon`: number of years in the horizon (e.g. 2). Required when target_mode is `calendar_horizon`. |
| **include_publication_year** | boolean | No | true | For `calendar_horizon`: if true, window is publication_year through publication_year + horizon_years (inclusive). If false, window is the next horizon_years full calendar years after publication year. |

---

### 2.4 `features` block (FeatureListsConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **numeric** | list of strings | No | `[]` | Names of numeric feature columns (must exist in the processed dataframe). |
| **categorical** | list of strings | No | `[]` | Names of categorical feature columns (must exist in the processed dataframe). |

Column order in the feature matrix is: numeric first, then categorical. This order must match what the preprocessor and any model that uses column indices (e.g. year_conditioned) expect.

---

### 2.5 `split` block (SplitConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **split_kind** | string | No | `"random"` | One of: `random`, `time`. `random` = shuffle and split by fraction. `time` = split by a time column (e.g. publication_year). |
| **test_size** | float | No | 0.2 | Fraction of data used as test set (0 < test_size < 1). Used for random split; for time split with explicit year boundaries it can still be set but train_year_end / test_year_start take precedence. |
| **random_state** | integer | No | 42 | Random seed for reproducibility (random split and model training). |
| **time_column** | string or null | **Yes** when `split_kind: time` | null | For `split_kind: time`: column name for time-based split (e.g. `"publication_year"`). Required when split_kind is `time`; config load fails otherwise. |
| **train_year_end** | integer or null | No | null | For time split: train set = rows with time_column ≤ this value. |
| **test_year_start** | integer or null | No | null | For time split: test set = rows with time_column ≥ this value. |

When both `train_year_end` and `test_year_start` are set, they define explicit boundaries; otherwise the last `test_size` fraction of rows (by time) is used as test.

---

### 2.6 `model` block (ModelConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **name** | string | No | `"baseline"` | Model name as registered in `models/registry.py` (e.g. `baseline`, `ridge`, `elastic_net`, `xgboost`, `extra_trees`, `hist_gradient_boosting`, `year_conditioned`, `hurdle`). Active benchmark models exclude experimental/non-standard entries (e.g. `tweedie`). |
| **params** | object | No | `{}` | Model-specific hyperparameters passed to the builder (e.g. XGBoost `n_estimators`, `max_depth`; ridge `alpha`). |

---

### 2.7 `evaluation` block (EvaluationConfig)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **metrics** | list of strings | No | `["rmse", "mae", "r2"]` | Metric names to compute (e.g. `rmse`, `mae`, `r2`). |

---

### 2.8 `benchmark` block (BenchmarkMetadataConfig) — required for current jobs

**For `train`, `evaluate`, and `run`**, the `benchmark` block is **required**. These values are written into the metrics JSON and used by `benchmark-analysis`. Omitting the block or leaving `benchmark_mode` / `model_family` empty causes load to fail with a migration hint. Inference from experiment name or dataset_id exists **only when reading older metrics artifacts** (legacy compatibility), not for authoring new configs.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **benchmark_mode** | string | Yes (non-empty) | — | Canonical benchmark mode: `representative_proxy`, `temporal_proxy`, `representative_h2`, or `temporal_h2`. |
| **model_family** | string | Yes (non-empty) | — | Label for report grouping, e.g. `trivial_baseline`, `linear_baseline`, `tree_model`, `hurdle_baseline`, `diagnostic_baseline`. |
| **is_diagnostic_model** | boolean | No | false | If true, the model is marked as diagnostic-only in comparison reports (e.g. year-conditioned baseline), not a primary comparator. |

---

### 2.9 `ablation` block (AblationConfig) — optional; required fields when present

When present, this run is treated as an ablation. **Both `name` and `features_removed` (non-empty list) are required**; config load fails otherwise. They are the single source of truth for ablation reports.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| **name** | string | Yes (non-empty) | — | Short ablation identifier (e.g. `no_publication_year`, `no_referenced_works_count`). |
| **features_removed** | list of strings | Yes (non-empty when ablation present) | — | List of feature names removed in this experiment. Written to metrics JSON and used by benchmark-analysis. |
| **ablation_type** | string or null | No | null | Optional. One of `coarse` (feature-group ablation) or `numeric_fine` (single numeric feature). Used in ablation review reports. |

---

## 3. Example: minimal experiment config (with required benchmark)

For `train` / `evaluate` / `run`, the `benchmark` block is required.

```yaml
experiment_name: my_experiment
task_type: regression
benchmark:
  benchmark_mode: representative_proxy
  model_family: linear_baseline
  is_diagnostic_model: false

data:
  processed_path: "data/processed/openalex_temporal_articles_1000.parquet"
  dataset_id: openalex_temporal_articles_1000

target:
  name: cited_by_count
  transform: log1p
  target_mode: proxy

features:
  numeric:
    - publication_year
    - referenced_works_count
    - authors_count
    - institutions_count
  categorical:
    - type
    - language
    - venue_name
    - primary_topic
    - open_access_is_oa

split:
  split_kind: random
  test_size: 0.2
  random_state: 42

model:
  name: ridge
  params: {}

evaluation:
  metrics:
    - rmse
    - mae
    - r2
```

---

## 4. Example: experiment config with benchmark and ablation

```yaml
experiment_name: xgb_temporal_h2_no_publication_year
task_type: regression
benchmark:
  benchmark_mode: temporal_h2
  model_family: tree_model
  is_diagnostic_model: false
ablation:
  name: no_publication_year
  features_removed: [publication_year]
  ablation_type: coarse

data:
  processed_path: "data/processed/openalex_temporal_articles_1000.parquet"
  dataset_id: openalex_temporal_articles_1000

target:
  name: citations_within_2_calendar_years
  target_mode: calendar_horizon
  source: counts_by_year
  horizon_years: 2
  include_publication_year: true
  transform: log1p

features:
  numeric:
    - referenced_works_count
    - authors_count
    - institutions_count
  categorical:
    - type
    - language
    - venue_name
    - primary_topic
    - open_access_is_oa

split:
  split_kind: time
  time_column: publication_year
  train_year_end: 2018
  test_year_start: 2019
  test_size: 0.2
  random_state: 42

model:
  name: xgboost
  params:
    n_estimators: 50
    max_depth: 4
    # ...

evaluation:
  metrics:
    - rmse
    - mae
    - r2
```

---

## 5. Validation

- Configs are loaded and validated with **Pydantic**. Invalid types or missing required fields will raise at load time.
- Data configs: `load_data_config(path)` → `DataConfig`.
- Experiment configs: `load_experiment_config(path)` → `ExperimentConfig` (permissive; use for report-only or legacy paths). For **train / evaluate / run** the CLI uses `load_current_experiment_config(path)`, which **requires** a valid `benchmark` block and valid `ablation` block when present; otherwise it raises with a migration hint.
- Semantic rules: `split_kind: time` requires non-empty `split.time_column`. `target_mode: calendar_horizon` requires `target.source` and `target.horizon_years`. Empty `benchmark_mode` or `model_family` when the benchmark block is present is rejected.
- Schema definitions live in `src/scholarly_outcome_prediction/settings.py`.
