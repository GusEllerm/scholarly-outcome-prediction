# Architecture

## Pipeline stages

1. **Fetch** — Request a reproducible sample of works from the OpenAlex API; save raw responses as JSONL in `data/raw/`.
2. **Prepare** — Load raw JSONL, normalize each work into a flat record (see [Normalization](#normalization)), write a single Parquet table to `data/processed/{dataset_name}.parquet`.
3. **Train** — Load processed Parquet, extract metadata features and target (no imputation here), drop rows with missing target, split train/test, fit a preprocessor (imputation + encoding) + model pipeline on training data only, save the pipeline to `artifacts/models/{experiment_name}.joblib`.
4. **Evaluate** — Load the same processed data and saved pipeline, reproduce the split, predict on the test set, compute metrics, write a **self-describing** JSON artifact to `artifacts/metrics/{experiment_name}.json` (metrics plus run metadata).

The **run** command runs: fetch → prepare → train (e.g. baseline + XGBoost) → evaluate both, using one data config and two experiment configs.

## Target semantics: proxy vs research target

- **Proxy target** (`target_mode: proxy`): Current bootstrap target, e.g. present-day cumulative `cited_by_count`. Used for development and benchmarking. Do not confuse with the intended long-term evaluation.
- **Research target** (`target_mode: research`): Future fixed-horizon scholarly outcome (e.g. citations at 3 years post-publication). Not fully implemented yet; config and code are structured so the repo can support both.

Always set `target_mode` in experiment configs and document which target a run used. Run artifacts record `target_mode` for interpretability.

## Config structure

- **Data config** (`configs/data/*.yaml`): `dataset_name`, `seed`, `sample_size`, `from_publication_date`, `to_publication_date`, `fields`, `output_path`.
  - **Debug** (`openalex_debug.yaml`): Small, fast; for smoke tests and local iteration.
  - **Pilot** (`openalex_pilot.yaml`): Broader date range and larger sample (e.g. 1000) for a more realistic first benchmark; `publication_year` varies.
  - **Sample** (`openalex_sample_100.yaml`): Original narrow sample (e.g. 2018-only, 100 works).
- **Experiment config** (`configs/experiments/*.yaml`): `experiment_name`, `task_type`, `data.processed_path`, `data.dataset_id`, `target.name` / `target.transform` / `target.target_mode`, `features.numeric` / `features.categorical`, `split.split_kind` / `split.test_size` / `split.random_state` (and optional `split.time_column` for future time-based split), `model.name` / `model.params`, `evaluation.metrics`.

Paths in configs are relative to the project root.

## Run artifact metadata

Every metrics JSON saved by `evaluate` (or `run`) includes run metadata so results are interpretable without reading the code. It records: `experiment_name`, `target_name`, `target_transform`, `target_mode`, `model_name`, `model_params`, `feature_numeric`, `feature_categorical`, `split_kind`, `split_test_size`, `split_random_state`, `train_size`, `test_size`, `dataset_id`, `run_id` (timestamp), plus the requested metrics (e.g. `rmse`, `mae`, `r2`).

## Evaluation and splitting

- **Current**: Random train/test split with fixed seed (`split_kind: random`). Reproducible and suitable for bootstrap.
- **Future**: The split layer accepts `split_kind` and optional `time_column` so that time-based splits (e.g. train on earlier years, test on later) and classification setups can be added without changing the high-level flow.

## Where data and artifacts are stored

| Location | Content |
|----------|--------|
| `data/raw/` | Raw OpenAlex API responses (JSONL). |
| `data/interim/` | Reserved for intermediate tables. |
| `data/processed/` | Feature-ready Parquet tables (one per dataset). |
| `artifacts/models/` | Fitted pipelines (preprocessor + model) as joblib files. |
| `artifacts/metrics/` | Evaluation metrics + run metadata as JSON. |
| `artifacts/figures/` | Reserved for plots. |

## Normalization

Raw OpenAlex works are mapped to a flat schema. **Metadata fields**: `openalex_id`, `title`, `publication_year`, `publication_date`, `type`, `language`, `cited_by_count`, `referenced_works_count`, `authors_count`, `institutions_count`, `venue_name`, `open_access_is_oa`, `primary_topic`. **Optional text-modality fields** (for future use): `abstract_text`, `fulltext_text`, `has_abstract`, `has_fulltext`, `fulltext_origin`. Missing or malformed fields are handled defensively (nulls or safe defaults). Training currently uses metadata only.

## Feature layer and modality

- **Metadata features**: Built by `build_metadata_features()` (or `build_feature_matrix()`). Select columns, validate presence, extract target, apply target transform. **No imputation or encoding**; that happens in the sklearn pipeline fit on the training split only.
- **Text features**: Reserved for future use (e.g. abstract or full-text embeddings). Placeholder structure exists so the codebase can later add `build_text_features()` and hybrid pipelines without refactoring the whole stack.

## Task abstraction

The repo separates **dataset**, **task**, and **model**. Experiment configs are task-aware (`task_type`, `target`, `evaluation.metrics`). Current supported task: citation regression (proxy target). Future tasks (e.g. citation-bin classification, venue prediction) can be added by extending task handling and evaluation without changing the fetch/prepare pipeline.

## How to add a new model

1. Implement a builder function or class that returns a scikit-learn–compatible regressor (e.g. in `src/scholarly_outcome_prediction/models/`).
2. Register it in `models/registry.py`: add a name and a callable that accepts `params` and returns the estimator.
3. Add a new experiment config YAML that sets `model.name` and `model.params`.
4. Run `train` and `evaluate` with that config (or extend the `run` flow if desired).

For the model to be classified correctly in benchmark comparison and ablation reports without relying on naming, add optional **`benchmark`** (and **`ablation`** if applicable) metadata to the experiment config; see **`docs/adding_a_model.md`** for the full checklist and field reference.

## How to add a new task

1. Extend experiment config and settings if needed (e.g. `task_type`, target semantics, metrics).
2. Ensure target extraction and evaluation are task-aware (e.g. classification metrics, different split logic).
3. Add or reuse data configs and document the intended use.
