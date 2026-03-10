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

## Targets: proxy and calendar-horizon

The benchmarks now support two target families:

- **Proxy target** (`target_mode: proxy`): present-day cumulative `cited_by_count` from the OpenAlex snapshot, with optional `log1p` transform. This is simple and always available, but mixes publication ages and does not fix a time window.
- **Calendar-horizon target** (`target_mode: calendar_horizon`): a **fixed-horizon citation** target derived from `counts_by_year` (citations bucketed by calendar year). You configure:
  - `horizon_years` (e.g. 2): how far out the **last** calendar year in the horizon extends, relative to publication year.
  - `include_publication_year` (true/false):
    - If `true`, the implementation sums years `publication_year` through `publication_year + horizon_years` (inclusive) – i.e. **`horizon_years + 1` calendar years** when including publication year.
    - If `false`, the window is the *next* `horizon_years` full calendar years after publication year (e.g. `publication_year+1` through `publication_year+horizon_years`).

These are **calendar-year** horizons, not month-level windows. Eligibility is enforced by requiring that the latest citation year observed in `counts_by_year` is at least `publication_year + horizon_years`; rows that do not have a fully observed horizon are excluded, and the exclusion counts are reported in run metadata and the target profile.

Representative and temporal pilots both have proxy and 2-year calendar-horizon examples:

- Representative: `baseline_representative`, `xgb_representative` (proxy) and `baseline_representative_h2`, `xgb_representative_h2` (2-year calendar-horizon).
- Temporal: `baseline_temporal`, `xgb_temporal` (proxy) and `baseline_temporal_h2`, `xgb_temporal_h2` (2-year calendar-horizon).

---

## Benchmark suite: baselines and models

Beyond the trivial (constant-mean) baseline, the suite includes stronger baselines so you can check whether XGBoost beats serious simple models.

| Model | Description | Config example |
|-------|-------------|----------------|
| **baseline** | Constant (training mean) | `baseline_temporal_h2.yaml` |
| **ridge** | Regularized linear regression on the same metadata features | `ridge_temporal_h2.yaml` |
| **year_conditioned** | Per–publication-year median (or mean) of the training target; unseen years get global median. Tests whether the model is mostly exploiting time/cohort. | `year_conditioned_temporal_h2.yaml` |
| **hurdle** | Two-stage: zero vs nonzero classifier, then Ridge on positive targets only. | `hurdle_temporal_h2.yaml`; `make run-temporal-h2-hurdle` |
| **xgboost** | Gradient-boosted trees. | `xgb_temporal_h2.yaml` |

**Makefile (temporal H2):**

- `make run-temporal-h2` — Full pipeline: data + **baseline** + **xgboost** (writes `baseline_temporal_h2`, `xgb_temporal_h2` metrics).
- `make run-temporal-h2-baselines` — Same data, **ridge** + **year_conditioned** (writes `ridge_temporal_h2`, `year_conditioned_temporal_h2` metrics). Run after `run-temporal-h2` so all four model metrics exist for comparison.
- `make run-temporal-h2-hurdle` — Same data, **hurdle** baseline (writes `hurdle_temporal_h2` metrics). Run after `run-temporal-h2`.

**Run the whole benchmark suite:**

```bash
make run-full-benchmark
```

This runs in order:

1. **Representative proxy** — `run-representative-pilot` (representative data, proxy target, baseline + xgboost).
2. **Representative H2** — `run-representative-h2` (same representative data, 2-year calendar-horizon, baseline + xgboost).
3. **Temporal proxy** — `run-temporal-pilot` (temporal data, proxy target, baseline + xgboost).
4. **Temporal H2** — `run-temporal-h2` (temporal data, 2-year calendar-horizon, baseline + xgboost).
5. **Temporal H2 baselines** — `run-temporal-h2-baselines` (ridge + year_conditioned on temporal H2).
6. **Temporal H2 hurdle** — `run-temporal-h2-hurdle` (hurdle baseline on temporal H2).
7. **Temporal H2 ablations** — `run-temporal-h2-ablations` (five ablation experiments).
8. **benchmark-analysis** — Writes `artifacts/reports/benchmark_comparison.json` and `.md`, and `ablation_review.json` and `.md`.

After this, the comparison report has rows for all four benchmark modes (with baseline and xgboost at least), and temporal_h2 also has ridge, year_conditioned, and hurdle. You can run individual steps if you only need a subset.


---

## Benchmark comparison and ablation review

After running one or more benchmark modes, a single command aggregates metrics and produces comparison reports.

### Unified benchmark comparison

- **Command:** `make benchmark-analysis` (or `scholarly-outcome-prediction benchmark-analysis`).
- **Input:** All `artifacts/metrics/*.json` files.
- **Output:** `artifacts/reports/benchmark_comparison.json` and `benchmark_comparison.md`.

The report has one row per **(benchmark mode, model)** with primary metrics (RMSE, MAE, R²), test zero-rate, and MAE on zero-target vs nonzero-target subsets. The four benchmark modes considered are:

- **representative_proxy** — Representative data, proxy target.
- **temporal_proxy** — Temporal split, proxy target.
- **representative_h2** — Representative data, 2-year calendar-horizon target.
- **temporal_h2** — Temporal split, 2-year calendar-horizon target.

If a combination was never run, it appears in a **Missing** list (no silent skips). This lets reviewers compare representative vs temporal and proxy vs H2 in one place.

### Metadata ablations

Ablations remove specific feature groups to see what signal the benchmark is using. Configs live in `configs/experiments/ablations/`:

| Ablation | Features removed |
|----------|------------------|
| `no_publication_year` | `publication_year` |
| `no_venue_name` | `venue_name` |
| `no_primary_topic` | `primary_topic` |
| `numeric_only` | All categorical features |
| `categorical_only` | All numeric features |

Each ablation is a separate experiment (e.g. `xgb_temporal_h2_no_publication_year`). Run them after temporal H2 data exists:

```bash
make run-temporal-h2          # data + baseline + xgboost
make run-temporal-h2-ablations   # train + evaluate each ablation config
make benchmark-analysis       # regenerates comparison + ablation review
```

**Ablation review** (`artifacts/reports/ablation_review.json` and `.md`) is produced by the same `benchmark-analysis` command. It lists each ablation run with metrics and **deltas vs the full XGBoost temporal H2 model**, plus a short interpretation (e.g. whether removing a feature group hurts or helps). If no ablation runs exist yet, the report shows a hint explaining how to populate it.

---

## Evaluation: zero-inflation and calibration/tail diagnostics

Metrics JSONs written by the pipeline include more than RMSE/MAE/R² when using the current evaluation path:

- **Zero-inflation** (`zero_inflation`): Test zero-rate and nonzero-rate; MAE and RMSE on the subset of rows with target = 0 and on the subset with target > 0. This makes it explicit whether performance is driven mostly by zero-target behaviour.
- **Calibration/tail** (`calibration_tail`): For regression we do **not** use classification-style calibration. Instead we store:
  - **By target decile:** For each decile of the actual target: count, mean actual, mean predicted, mean residual, MAE (to see over/under-prediction by bucket).
  - **Top quantiles:** MAE and RMSE on the top 90th, 95th, 99th percentiles of the target (tail performance).

These support questions like: *Is the model well-behaved on the upper tail?* and *How much of the overall metric comes from zero vs nonzero targets?*
