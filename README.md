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

Initial experiement:

> Predict citation count from OpenAlex metadata using a small, reproducible sample and a baseline + XGBoost model.

## Current scope

v0.1 is focused on a single end-to-end path:

1. fetch a reproducible sample of works from OpenAlex
2. save the raw responses locally
3. normalize records into a flat tabular dataset
4. build a first-pass metadata feature set
5. train a baseline model and an XGBoost model
6. evaluate the models
7. save metrics and model artifacts

This is a **prototype**.

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
│   │   └── openalex_sample_100.yaml
│   └── experiments/
│       ├── baseline_regression.yaml
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

### Candidate first-pass features

Numeric:

- `publication_year`
- `referenced_works_count`
- `authors_count`
- `institutions_count`

Categorical:

- `type`
- `language`
- `venue_name`
- `primary_topic`
- `open_access_is_oa`

Target:

- `cited_by_count`, transformed using `log1p`

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
scholarly-outcome-prediction fetch --config configs/data/openalex_sample_100.yaml
scholarly-outcome-prediction prepare --config configs/data/openalex_sample_100.yaml
scholarly-outcome-prediction train --config configs/experiments/xgb_regression.yaml
scholarly-outcome-prediction evaluate --config configs/experiments/xgb_regression.yaml
scholarly-outcome-prediction run --config configs/experiments/xgb_regression.yaml
```

Synonymous Makefile:

```bash
make install
make lint
make format
make test
make run-example
```

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

## Roadmap

### Phase 1: bootstrap

- [ ] OpenAlex fetcher
- [ ] normalization pipeline
- [ ] metadata feature builder
- [ ] baseline regressor
- [ ] XGBoost regressor
- [ ] evaluation + saved metrics
- [ ] tests + CLI

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
