# scholarly-outcome-prediction

An initial pipeline for predicting scholarly outcomes from linked research metadata.

## Overview

This repository is a starter for building and comparing models that predict outcomes for scholarly works using structured metadata.

The long-term goal is to support multiple outcome prediction tasks, such as:

- citation count prediction
- venue prediction
- patent citation prediction
- downstream clinical or translational impact proxies
- other metadata-derived scholarly outcome tasks

Initial experiment:

> Predict citation count from OpenAlex metadata using a small, reproducible sample and baseline + stronger baselines + XGBoost.

## Current scope

- **Metadata-only**: Features are metadata-derived (publication year, counts, venue, topic, etc.). Text (abstracts, full text) is not used in training yet; the schema and feature layer are set up for future text modalities.
- **Two target modes**: (1) **Proxy** — present-day cumulative `cited_by_count`; (2) **Calendar-horizon** — citations within a fixed number of full calendar years derived from OpenAlex `counts_by_year`. Both are configurable; run artifacts record target mode and semantics.
- Single end-to-end path: fetch → prepare → build metadata features → split → preprocess (imputation + encoding on train only) → train → evaluate → save self-describing metrics and run metadata.

This is a **prototype** with a path to broader tasks and evaluation.

## Project principles

This repository is being built around a few non-negotiables:

- **Reproducibility first**: config-driven runs, fixed seeds, pinned dependencies, saved artifacts
- **Extensible**: new models and new outcome tasks should be straightforward to add
- **Follow Observability Principles**: it should be easy to see how outputs were produced from source data

## Repository layout

```text
scholarly-outcome-prediction/
├── README.md
├── LICENSE
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── configs/
│   ├── data/
│   │   ├── openalex_debug.yaml      # small, fast (smoke tests)
│   │   ├── openalex_pilot.yaml      # broader, ~1k (first benchmark)
│   │   └── openalex_sample_100.yaml
│   └── experiments/
│       ├── baseline_regression.yaml
│       ├── ridge_regression.yaml
│       └── xgb_regression.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── figures/
├── notebooks/
│   └── exploratory_openalex.ipynb
├── scripts/
│   └── run_experiment.py
├── src/
│   └── scholarly_outcome_prediction/
│       ├── __init__.py
│       ├── cli.py
│       ├── settings.py
│       ├── logging_utils.py
│       ├── acquisition/
│       │   ├── __init__.py
│       │   ├── openalex_client.py
│       │   └── fetch.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── normalize.py
│       │   └── split.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── build_features.py
│       │   └── preprocess.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── baseline.py
│       │   └── xgboost_model.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── report.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           └── seeds.py
├── tests/
│   ├── conftest.py
│   ├── test_normalize.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_smoke_pipeline.py
└── docs/
    ├── architecture.md
    └── experiment-log.md
```

## Technology choices

The initial stack is deliberately conservative:

- **Python 3.11**
- **uv** for dependency management and lockfiles
- **pyproject.toml** for packaging and project metadata
- **pandas / numpy** for tabular data handling
- **scikit-learn** for preprocessing, evaluation, and baselines
- **XGBoost** for the first non-trivial model
- **PyYAML + Pydantic** for config loading and validation
- **Typer** for a simple CLI
- **pytest + ruff + pre-commit** for code quality and testing

### Two benchmark modes

The pipeline supports two distinct pilot modes:

- **Representative pilot** — For broad laptop-scale benchmarking and sanity checks on feature usefulness. Uses article-only OpenAlex data with **within-year random sampling** (OpenAlex `sample` + `seed` per year) so the corpus is not biased toward highly cited works. Train/test split is **random**. Validation can fail if the corpus looks elite-only (e.g. median citations above threshold).
- **Temporal pilot** — For temporal generalization: train on earlier publications, evaluate on later ones. Uses stratified-by-year fetch (cursor-based, no random) so the dataset has real year spread, then **time-based split** with explicit year boundaries (e.g. train on 2015–2018, test on 2019–2020).

Representative and temporal are separate configs and Makefile targets so behavior is explicit and reproducible. See [Benchmark modes](docs/benchmark_modes.md) for details.

### Data configs

- **Debug** (`configs/data/openalex_debug.yaml`): Small sample, fast; for smoke tests and local iteration.
- **Pilot** (`configs/data/openalex_pilot.yaml`): Broader date range and larger sample (~1000); legacy (no stratified fetch).
- **Representative** (`openalex_representative_articles_1000.yaml`): Article-only, `stratify_by_year: true`, 2015–2020; for representative benchmarking with random split.
- **Temporal** (`openalex_temporal_articles_1000.yaml`): Article-only, `stratify_by_year: true`, 2015–2020; for time-based evaluation.
- **Sample** (`openalex_sample_100.yaml`): Original narrow sample (e.g. 2018-only, 100 works).

### Metadata features (current)

Numeric: `publication_year`, `referenced_works_count`, `authors_count`, `institutions_count`.  
Categorical: `type`, `language`, `venue_name`, `primary_topic`, `open_access_is_oa`.  
Target: **Proxy** — `cited_by_count`; **Calendar-horizon** — sum of citations from `counts_by_year` over a configurable number of full calendar years. Both support optional `log1p` transform.  
Imputation and encoding are done only in the sklearn pipeline fit on the training split (no leakage from feature building).

### Target modes: proxy vs calendar-horizon

The pipeline supports two target families; do not confuse them.

- **Proxy** (`target_mode: proxy`): Uses current cumulative `cited_by_count` from the snapshot. Simple and always available, but mixes publication ages and does not fix a time window — useful for debugging and comparison only.
- **Calendar-horizon** (`target_mode: calendar_horizon`): Derives the target from OpenAlex `counts_by_year` (citations bucketed by calendar year). You choose:
  - **Horizon length** (`horizon_years`, e.g. 2): how many full calendar years to include.
  - **Include publication year** (`include_publication_year: true/false`): if `true`, the window is publication year through publication year + (horizon_years − 1); if `false`, the window is the *next* `horizon_years` full calendar years after publication year.

**Important:** These are **calendar-year** targets, not exact month-level windows. We use honest naming (e.g. `citations_within_2_calendar_years`, `citations_in_next_2_calendar_years`) and do *not* claim “citations after exactly 24 months”.

**Horizon eligibility:** A row is only used when the requested horizon is fully observed in the data. For example, if the latest citation year in the snapshot is 2026, a paper published in 2025 with a 2-year horizon (including publication year) is **excluded**, because we do not yet have citation counts through 2026. The pipeline computes a global `max_available_citation_year` from `counts_by_year`, excludes ineligible rows, and reports how many were dropped (e.g. in run metadata and metrics).

**How to run calendar-horizon benchmarks (2-year example):**

```bash
make run-representative-h2   # representative data, 2-year calendar-horizon target
make run-temporal-h2        # temporal split, 2-year calendar-horizon target
```

**Limitations of the calendar-horizon approach:** (1) Granularity is calendar years, not exact months or days. (2) Eligibility depends on the snapshot’s latest citation year, so newer papers are excluded when the horizon is not yet complete. (3) We do not yet expand citing papers via `cited_by_api_url`; the target is derived only from the work’s own `counts_by_year`.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd scholarly-outcome-prediction
```

### 2. Install dependencies with `uv`

```bash
uv sync
```

### 3. Activate the environment if needed

```bash
source .venv/bin/activate
```

### 4. Set environment variables

Copy the example file:

```bash
cp .env.example .env
```

If using the OpenAlex polite pool, set your email address:

```env
OPENALEX_MAILTO=your.email@example.org
```

## Running the pipeline

The intended CLI shape is:

```bash
# Step by step:
scholarly-outcome-prediction fetch --config configs/data/openalex_sample_100.yaml
scholarly-outcome-prediction prepare --config configs/data/openalex_sample_100.yaml
scholarly-outcome-prediction train --config configs/experiments/baseline_regression.yaml
scholarly-outcome-prediction train --config configs/experiments/xgb_regression.yaml
scholarly-outcome-prediction evaluate --config configs/experiments/baseline_regression.yaml
scholarly-outcome-prediction evaluate --config configs/experiments/xgb_regression.yaml

# Or full pipeline in one go:
scholarly-outcome-prediction run \
  --data-config configs/data/openalex_sample_100.yaml \
  --baseline-config configs/experiments/baseline_regression.yaml \
  --xgb-config configs/experiments/xgb_regression.yaml
```

Makefile commands:

```bash
make install
make lint
make test

# Representative pilot (stratified fetch, random split)
make run-representative-pilot

# Temporal pilot (stratified fetch, train 2015–2018 / test 2019–2020)
make run-temporal-pilot

# Calendar-horizon targets (2-year window from counts_by_year)
make run-representative-h2
make run-temporal-h2

# Validate representative or temporal dataset
make validate-representative-pilot
make validate-temporal-pilot

# Regenerate diagnostics (stamped with dataset/run ID)
make profile-representative-pilot
make profile-temporal-pilot
```

Other targets:

```bash
make install
make lint
make format
make test
make run-example
```

## Run artifacts

Each evaluation run writes a **self-describing** JSON file under `artifacts/metrics/` that includes: experiment name, target name/transform/mode, model name and params, feature lists, split settings, train/test sizes, dataset id, run timestamp, and the requested metrics (e.g. RMSE, MAE, R²). For **calendar-horizon** runs, metrics JSON also includes target semantics description, eligibility counts, target zero-rate, and how missing/empty `counts_by_year` is handled. See `docs/architecture.md` for the full list.

## Project diagnostics and transparency

The repo includes a **diagnostics** pass to keep the pipeline inspectable and debuggable.

- **Report:** `docs/refinement_debug_report.md` — describes current pipeline flow, task definition, features, data quality, preprocessing and leakage, evaluation design, and prioritized next steps. Read this for a precise picture of what the code does and where it diverges from the intended scholarly-outcome task.
- **Regenerating diagnostics:** From the project root, run:
  ```bash
  uv run python scripts/generate_diagnostics.py
  ```
  Optional: `--processed path/to/processed.parquet` and `--dataset-id ID` to profile a specific dataset (default: `data/processed/openalex_pilot.parquet`). This writes under `artifacts/diagnostics/`:
  - `component_inventory.json` — major modules and critical path for `run` (design-scoped)
  - `pipeline_trace_design.json` — design-scoped step trace. When you run the pipeline (`make run-representative-pilot` or any `make run-*`), the CLI writes **both** the run-scoped `pipeline_trace.json` and all other diagnostics (profile, artifact audit, design trace, etc.) in one go, so you get a full set without running this script.
  - `dataset_profile.json` — row count, publication year, **dataset-level** citation stats (`cited_by_count`), missingness, categorical tops (dataset-scoped; uses canonical stats shared with validation)
  - `target_profile.json` — **target-level** report for calendar-horizon runs: eligibility filtering, untransformed/transformed target distribution, zero-rate, and empty/missing `counts_by_year` diagnostics (run-scoped). Companion `target_profile.md` for human reading.
  - `missingness_summary.csv` — per-column missing count and percent
  - `feature_usage_report.json` — normalized schema vs config feature lists, used/unused, leakage-risky columns
  - `run_artifact_audit.json` — which metrics/models exist, metadata completeness, baseline vs xgb agreement (run- or dataset-scoped)
  - `preprocessing_leakage_audit.json` — verified preprocessing facts vs design caveats, leakage risk (design-scoped)
- **Package:** `src/scholarly_outcome_prediction/diagnostics/` provides the functions used by the script; you can call them from notebooks or other scripts to inspect configs or datasets.

### Report scopes

Every major diagnostic JSON includes `report_scope`, `report_name`, and `generated_at` so you can tell what the file describes:

- **`run`** — Describes one execution: data config used, experiments run, stages completed, consistency checks. Example: `pipeline_trace.json` after `uv run run ...`, or validation JSON when validation was run as part of a pipeline.
- **`dataset`** — Describes a processed dataset (profile, feature usage, or validation when run standalone). Not tied to a specific run instance.
- **`design`** — Describes the codebase/architecture (component inventory, static pipeline trace, preprocessing audit). Same for every run; findings are from static code inspection, not from observing a specific execution.

### Identifiers: run_id vs dataset_id vs report_id / audit_id

- **`run_id`** — Used only in **run-scoped** reports. It identifies a specific execution instance (e.g. an ISO timestamp from when the pipeline run started). Do not use it for dataset names or report grouping in non-run reports.
- **`dataset_id`** / **`source_dataset_id`** — Identify the dataset (e.g. processed file stem, or `dataset_name` from the data config). Present in run- and dataset-scoped reports when applicable.
- **`report_id`** — Identifies a report artifact (e.g. `{source_dataset_id}_dataset_validation`). Used in dataset-scoped reports so the report is uniquely labeled without implying a run instance.
- **`audit_id`** — Identifies an audit artifact (e.g. artifact audit). Used in dataset-scoped or design-scoped audits instead of `run_id`.

Design-scoped reports do not include `run_id`. They may include a `config_paths_note` or `design_note` stating that config paths are not applicable or that findings are inferred from code, not from a runtime trace.

### Dataset profile vs target profile

Do not confuse **dataset-level** and **target-level** reporting.

- **Dataset profile** (`dataset_profile.json`) describes the **processed dataset** as stored on disk: row count, publication year range, **current cumulative** `cited_by_count` distribution, feature missingness, venue/topic/language diversity. It does **not** apply eligibility filtering or describe the actual supervised-learning label used in an experiment.
- **Target profile** (`target_profile.json`, generated for calendar-horizon runs) describes the **modeling target**: target config (name, mode, source, horizon years, include publication year, transform), eligibility summary (rows raw vs eligible vs excluded for incomplete horizon), **untransformed and transformed target distribution**, zero-target rate, and explicit handling of missing/empty `counts_by_year`. It also reports how many rows had empty/missing `counts_by_year` and, among those, how many had `cited_by_count == 0` vs `cited_by_count > 0`, so reviewers see that empty `counts_by_year` is not equivalent to “no citations.”

**Why `cited_by_count` ≠ calendar-horizon target:** Dataset-level `cited_by_count` is the snapshot’s cumulative citation count. The calendar-horizon target is the sum of citations over a fixed number of **calendar years** from `counts_by_year`. They are different quantities; the target profile reports the one actually used for training and evaluation.

**How empty/missing `counts_by_year` is handled:** Missing or empty `counts_by_year_json` is explicitly treated as a **zero yearly-count series**: each year in the horizon contributes 0, so the target value for such rows is 0. This is documented in `features/targets.py` and summarized in the target profile. Calendar-horizon targets remain an **approximation** (calendar-year granularity, not exact month-level windows; eligibility depends on the snapshot’s latest citation year).

### Validation severity

Dataset validation outputs include a `messages` list with `severity` per message:

- **`error`** — Validation failed (e.g. row count below minimum, venue missingness too high).
- **`warning`** — Concern (e.g. single year only, high median citations for representative mode).
- **`informational`** — Note only.
- **`expected`** — Matches config (e.g. article-only corpus when `work_types: [article]`); not a warning.

When the data config sets `work_types: [article]`, a single-type (article) corpus is reported as expected, not as “single type only”.

### Verifying a run

1. After `make run-representative-pilot` (or any `make run-*`), the CLI writes a **run-scoped** trace to `artifacts/diagnostics/pipeline_trace.json` and generates all other diagnostics in the same directory (profile, artifact audit, design trace, etc.). Check `pipeline_trace.json`: `report_scope` should be `run`; `data_config` should list the fetch controls (work_types, stratify_by_year, use_random_sample, effective_sampling_strategy); `experiments` should list each experiment’s config path, target, split_kind, and effective_processed_path.
2. Check `artifacts/reports/{dataset_name}_dataset_validation.json`: when produced by the pipeline it is run-scoped (has `run_id`); when produced by `validate` only it is dataset-scoped (has `report_id`, no `run_id`). Check `passed`, `errors`, `warnings`, and `messages` (with severity). For article-only representative runs, you should see an `expected` message, not a “single type only” warning.
3. Check `artifacts/diagnostics/run_artifact_audit.json`: when produced by `generate_diagnostics` it is dataset-scoped (`audit_id`, no `run_id`). Check `metrics_found` / `model_found`, `baseline_xgb_agreement`, and `consistency_checks` in the run-scoped pipeline trace.
4. Cross-check: pipeline trace `consistency_checks` (dataset_id_match, processed_path_match, validation_input_match, baseline_xgb_metadata_match, artifacts_present) and `config_paths` (actual resolved paths). Metrics JSONs should have the same `effective_dataset_id` and `effective_processed_path` as the run.

## Data and artifact conventions

### Raw data

Raw OpenAlex API responses should be stored in:

```text
data/raw/
```

These should remain as close to the source responses as practical, ideally in JSONL format.

### Intermediate data

Normalized but not final representations should live in:

```text
data/interim/
```

### Processed data

Feature-ready tabular datasets should live in:

```text
data/processed/
```

Parquet is the preferred format for processed datasets.

## Current limitations

This first version is limited. It does **not** yet aim to include:

- large-scale corpora
- text embeddings or abstract-based models
- graph neural networks or citation graph models
- cloud orchestration
- hyperparameter search infrastructure
- experiment tracking platforms like MLflow
- multi-task training
- patent or clinical-outcome joins

Those are future layers once the basic research pipeline is stable.

**Note for macOS users:** XGBoost may require the OpenMP runtime. If you see an error loading `libxgboost.dylib`, run `brew install libomp`. Tests that require XGBoost will be skipped if the library cannot be loaded.

## Roadmap

### Phase 1: bootstrap

- [x] OpenAlex fetcher
- [x] normalization pipeline
- [x] metadata feature builder
- [x] baseline regressor
- [x] XGBoost regressor
- [x] evaluation + saved metrics
- [x] tests + CLI

### Phase 2: stronger experiments

- [ ] larger datasets
- [ ] time-aware splits
- [ ] classification targets and citation bins
- [ ] additional model families
- [ ] better feature reporting and importances

### Phase 3: broader outcomes

- [ ] venue prediction
- [ ] patent citation prediction
- [ ] downstream translational impact proxies
- [ ] multi-task or ensemble experimentation

### Phase 4: scale and reproducibility hardening

- [ ] containerized execution
- [ ] cloud training paths
- [ ] richer experiment tracking
- [ ] benchmark datasets and published releases

## Development expectations

This repo should stay disciplined.

Core expectations:

- keep notebooks out of core logic
- keep configs explicit
- prefer small, composable modules
- add tests when adding pipeline stages
- avoid premature abstractions
- make failures easy to diagnose

## License

Choose an open-source license appropriate for the project, such as MIT or Apache-2.0.

## Status

Early bootstrap stage. The architecture is being set up before scaling to broader outcome prediction tasks.
