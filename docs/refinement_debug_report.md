# Refinement and Debug Report — Scholarly Outcome Prediction

**Generated:** diagnostic pass over the current codebase.  
**Purpose:** give an external reviewer a precise, evidence-based picture of what the pipeline does, where it aligns with the intended task, and where it diverges.

---

## 1. Executive summary

### What the repo currently does

The repository implements a **single end-to-end path**: fetch a sample of works from the OpenAlex API → normalize to a flat table → build **metadata-only** features and target → random train/test split → fit a preprocessor (imputation + one-hot) and a regressor on the training set only → evaluate on the test set → save metrics and run metadata as JSON. The **current task** is **regression** with target **cited_by_count** (present-day cumulative), treated explicitly as a **proxy target** in config and artifacts. No text (abstract/full text) is used. Models supported: baseline (mean), median_baseline, ridge, xgboost.

### What improved since the initial bootstrap

- **Feature/preprocessing boundary:** Feature building no longer imputes or encodes; all imputation and encoding happen in the sklearn pipeline fit on the training split only (no leakage from whole-dataset statistics).
- **Target semantics:** Config and run artifacts distinguish **proxy** vs **research** target via `target_mode`; current runs are explicitly proxy.
- **Run metadata:** Each metrics JSON is self-describing (experiment name, target name/transform/mode, model name/params, feature lists, split settings, train/test sizes, dataset_id, run_id).
- **Data configs:** Debug (small/fast) and pilot (broader dates, larger N) configs exist; OpenAlex cursor paging was fixed so pilot can fetch 1000 works.
- **Schema:** Normalized schema includes optional text-modality fields (abstract_text, fulltext_text, has_abstract, etc.) for future use; they are not used in training.
- **Task/model separation:** Experiment configs separate dataset path, task type, target, features, model, and evaluation; the codebase is structured for multiple tasks later.

### Biggest methodological weaknesses still present

1. **Proxy target only:** The target is present-day cumulative citations, not a fixed-horizon outcome. Evaluation is therefore not directly comparable to an intended “citations at 3 years” research target.
2. **Random split only:** Split is random, not time-based. For citation prediction, a time-based split (e.g. train on earlier years, test on later) would better match real use and avoid temporal leakage.
3. **Possible corpus bias:** The pilot dataset (1000 works, 2015–2020) has a very high minimum citation count (e.g. ~4k in the profiled run). That suggests the OpenAlex sample may be biased toward highly cited works; the benchmark may not be representative of the full distribution.
4. **dataset_id vs actual data:** Run metadata records `dataset_id` from the **experiment** config, not from the data config used for fetch. If a user runs the pipeline with pilot data but experiment configs still point at `openalex_sample_100`, the artifact will claim `dataset_id: openalex_sample_100` even though the actual data was pilot.
5. **venue_name 100% missing in pilot:** In the profiled pilot data, `venue_name` is entirely missing (normalization or API response). That makes the “venue_name” feature degenerate for that corpus and may hurt model utility.

### Recommended next actions (priority order)

1. **Align artifact with actual data:** Either derive `dataset_id` from the data config used in `run` (or from the processed file path) and write it into run metadata, or document clearly that `dataset_id` is per-experiment-config and may not match the fetch.
2. **Investigate pilot corpus construction:** Confirm why citation minimum is so high and whether the OpenAlex query/sort introduces selection bias; consider explicit sampling or filters for a more representative benchmark.
3. **Fix or document venue extraction:** Establish why `venue_name` is all null in pilot and fix normalization or document that venue is unavailable for this dataset.
4. **Add time-based split:** Implement `split_kind: time` so that evaluation can use a temporal split when appropriate.
5. **Plan fixed-horizon target:** When data or joins are available, add a research target (e.g. citations at 3 years) and keep proxy for development only.

---

## 2. Repository component inventory

Major modules and their role (see also `artifacts/diagnostics/component_inventory.json`):

| Module | Role |
|--------|------|
| `cli` | Typer app; commands: fetch, prepare, train, evaluate, run. `run` calls `run_pipeline_from_configs` with one data config and two experiment configs. |
| `settings` | Pydantic models for DataConfig, ExperimentConfig (split, target, features, model, evaluation); `load_data_config`, `load_experiment_config`. |
| `acquisition.fetch` | `fetch_and_save`: loads data config, calls OpenAlex client, writes raw JSONL. |
| `acquisition.openalex_client` | `fetch_works_page`, `fetch_works_sample`: cursor-based paging (starts with `cursor="*"`), date filter. |
| `data.schemas` | `NormalizedWork` Pydantic model; extraction from raw OpenAlex work (including optional text fields). |
| `data.normalize` | `normalize_work`, `normalize_works_to_dataframe`; uses `NORMALIZED_COLUMNS`. |
| `data.split` | `train_test_split_df`; random split via sklearn; `split_kind="time"` not implemented. |
| `features.build_features` | `build_metadata_features` / `build_feature_matrix`: select columns, validate, extract target, apply log1p; **no imputation/encoding**. |
| `features.preprocess` | `build_preprocessor`: ColumnTransformer (numeric median impute, categorical fill + one-hot). Fitted only on train in pipeline. |
| `models.registry` | `get_model_builder`, `list_models`; registers baseline, median_baseline, ridge, xgboost. |
| `evaluation.report` | `build_run_metadata`, `save_metrics` (metrics + run metadata), `save_model_pipeline`, `load_model_pipeline`. |
| `evaluation.metrics` | `compute_metrics`: rmse, mae, r2 (on provided arrays). |
| `utils.io` | load_yaml, load_jsonl, save_jsonl, read_parquet, write_parquet, save_json, load_json. |
| `utils.seeds` | `set_global_seed` for reproducibility. |

**Critical path for `run`:** CLI → `run_pipeline_from_configs` → load_data_config + load_experiment_config (×2) → fetch_and_save → load_jsonl, normalize_works_to_dataframe, write_parquet → for each experiment config: read_parquet, build_feature_matrix, dropna(target), train_test_split_df, build_preprocessor, get_model_builder, Pipeline.fit, save_model_pipeline, compute_metrics, build_run_metadata, save_metrics.

---

## 3. Observed pipeline flow

Exact order of operations (see `artifacts/diagnostics/pipeline_trace.json`):

1. **Resolve root:** `root = data_config_path.resolve().parents[2]`.
2. **Load configs:** `load_data_config(data_config_path)`, `load_experiment_config(baseline_config_path)`, `load_experiment_config(xgb_config_path)`.
3. **Fetch:** `fetch_and_save(root / data_cfg.output_path, data_cfg.sample_size, data_cfg.from_publication_date, data_cfg.to_publication_date, data_cfg.seed)`. Writes raw JSONL to `data_cfg.output_path`.
4. **Prepare:** `load_jsonl(out_path)` → `normalize_works_to_dataframe(records)` → `write_parquet(df, root/data/processed/{data_cfg.dataset_name}.parquet)`.
5. **Per experiment (baseline, then xgb):**
   - `set_global_seed(cfg.split.random_state)`
   - `read_parquet(processed_path)` — **note:** `processed_path` comes from **that experiment’s** `cfg.data.processed_path`, which can differ from the file just written if the config points to another dataset.
   - `build_feature_matrix(df, num_feat, cat_feat, target_name, target_transform)` → X, y
   - `concat(X, y)`, `dropna(subset=[target])`
   - `train_test_split_df(full, test_size, random_state, split_kind, time_column)` → train_df, test_df
   - Extract X_train, y_train, X_test, y_test
   - `build_preprocessor(num_feat, cat_feat)`, `get_model_builder(cfg.model.name)(params=cfg.model.params)`
   - `Pipeline([("preprocessor", preprocessor), ("model", model)]).fit(X_train, y_train)`
   - `save_model_pipeline(pipe, artifacts/models/{experiment_name}.joblib)`
   - `compute_metrics(y_test, pipe.predict(X_test), cfg.evaluation.metrics)`
   - `build_run_metadata(...)`, `save_metrics(metrics, path, run_metadata)` → `artifacts/metrics/{experiment_name}.json`

**Configs read:** One data config YAML (dataset_name, seed, sample_size, dates, output_path). Two experiment config YAMLs (experiment_name, task_type, data.processed_path, data.dataset_id, target.*, features.*, split.*, model.*, evaluation.*).

**Outputs:** One raw JSONL, one processed Parquet (from data config), two model joblibs, two metrics JSONs (when running baseline + xgb).

---

## 4. Current benchmark / task definition

- **Task:** Regression (predict a single continuous target).
- **Target variable:** `cited_by_count` (present-day cumulative citation count from OpenAlex).
- **Target transform:** `log1p` applied in `build_metadata_features` before split; metrics (RMSE, MAE, R²) are computed **on the transformed scale** (log1p(citations)).
- **Target mode:** `target_mode: proxy` in all current experiment configs; run artifacts record this. The repo does not implement a fixed-horizon “research” target yet.
- **Model families:** baseline (training mean), median_baseline (training median), ridge (sklearn Ridge), xgboost (XGBRegressor). All are regressors; no classification or ranking is wired.

---

## 5. Feature and schema analysis

**Normalized fields (from `data/normalize.py`):** openalex_id, title, publication_year, publication_date, type, language, cited_by_count, referenced_works_count, authors_count, institutions_count, venue_name, open_access_is_oa, primary_topic, abstract_text, fulltext_text, has_abstract, has_fulltext, fulltext_origin.

**Used as features (all experiment configs):**  
Numeric: publication_year, referenced_works_count, authors_count, institutions_count.  
Categorical: type, language, venue_name, primary_topic, open_access_is_oa.

**Unused metadata columns (in schema but not in any feature list):** openalex_id, title, publication_date. (Optional text fields are also unused.)

**Leakage-prone columns (if ever used as features):** cited_by_count (target), title, openalex_id, publication_date. None of these appear in current feature lists; see `artifacts/diagnostics/feature_usage_report.json`.

**Text support:** Text is not used. Schema has abstract_text, fulltext_text, has_abstract, has_fulltext, fulltext_origin; training uses only the metadata features above. To support abstract/full text later: add a text feature builder, optional text columns in the feature matrix, and a preprocessor branch or separate pipeline for text (e.g. embeddings); config would need to specify which text fields to use.

---

## 6. Data and corpus diagnostics

Profiled dataset: `data/processed/openalex_pilot.parquet` (as in default diagnostics run).

| Metric | Value |
|--------|--------|
| Row count | 1000 |
| Publication year range | 2015–2020 |
| Publication year distribution | 2015: 184, 2016: 187, 2017: 207, 2018: 139, 2019: 111, 2020: 172 |
| cited_by_count min | 4396 |
| cited_by_count max | 801217 |
| cited_by_count mean | ~10232 |
| cited_by_count median | ~6476 |
| cited_by_count missing | 0 |

**Missingness (summary):** Language has negligible missing; venue_name, fulltext_text, has_fulltext, fulltext_origin are 100% missing; abstract_text / has_abstract ~65% missing. See `artifacts/diagnostics/dataset_profile.json` and `missingness_summary.csv`.

**Categorical summaries:** type dominated by article (665), then review (176), book (62), etc.; language dominated by en (987); primary_topic has many distinct values (e.g. Advanced Neural Network Applications, Genomics, COVID-19, etc.). venue_name has no non-null values in this profile, so categorical_tops for venue is empty.

**Suspicious signs:** Minimum citation count ~4k suggests the OpenAlex sample is not a random draw from all works in the date range; it may be sorted or filtered in a way that favors highly cited papers. That would make the benchmark easier and not representative of the long tail.

**Dataset ID consistency:** baseline_regression.json records `dataset_id: openalex_sample_100` with train_size 800, test_size 200 (1000 total); xgb_regression.json records `dataset_id: openalex_pilot` with the same sizes. So at least one run mixed configs: baseline was run with a config pointing at sample_100 (or an older run), while xgb was run with pilot. The pipeline does not overwrite `dataset_id` from the data config used for fetch; it uses the experiment config’s `data.dataset_id`.

---

## 7. Preprocessing and leakage analysis

**Where imputation happens:** In `features/preprocess.py`, inside the ColumnTransformer: numeric columns use `SimpleImputer(strategy="median")`, categorical columns use a `FunctionTransformer` that fills missing with `"__missing__"` and then `OneHotEncoder`. This transformer is part of the sklearn Pipeline and is **fit only on (X_train, y_train)** in the CLI/train flow.

**Where encoding happens:** Same preprocessor: one-hot encoding for categorical features after the fill step. Fit on train only.

**Split vs preprocessing:** Split is performed **before** any fit: `train_test_split_df(full, ...)` → then `Pipeline.fit(X_train, y_train)`. So test set is never used for fitting the preprocessor or the model.

**Target in features:** `cited_by_count` is not in any experiment’s feature list; it is only the target. No target leakage from column selection.

**Leakage risk assessment:** **Low** for the current design: feature building only selects and transforms the target; imputation and encoding are fit on the training set only; split is before fit. See `artifacts/diagnostics/preprocessing_leakage_audit.json`.

**Stale/metadata risk:** `dataset_id` in the metrics JSON is taken from the experiment config, not from the data config used in `run`. If a user runs with pilot data but leaves `data.processed_path` / `data.dataset_id` pointing at sample_100 in one of the experiment configs, the artifact will be misleading (wrong dataset_id and possibly wrong processed_path if they didn’t update it).

---

## 8. Evaluation and artifact review

**Split design:** Random split via `sklearn.model_selection.train_test_split` with `test_size=0.2`, `random_state=42`, `split_kind="random"`. Time-based split is not implemented (`split_kind="time"` raises NotImplementedError).

**Metrics saved:** rmse, mae, r2 (on the transformed target scale, log1p(cited_by_count)).

**Run metadata in metrics JSON:** experiment_name, target_name, target_transform, target_mode, model_name, model_params, feature_numeric, feature_categorical, split_kind, split_test_size, split_random_state, train_size, test_size, dataset_id, run_id (timestamp). All required keys are present in the audited files; see `artifacts/diagnostics/run_artifact_audit.json`.

**Consistency:** As noted, dataset_id and processed_path are per-experiment config; they do not automatically reflect the data config used in a given `run`. So run metadata can be inconsistent with the actual corpus (e.g. baseline_regression.json with dataset_id openalex_sample_100 vs actual 1000-row pilot data).

---

## 9. Readiness for text and richer outcome tasks

**What blocks title/abstract/full-text use today:** No feature builder consumes text columns; the preprocessor only handles the current numeric and categorical lists. Schema and normalized columns already have optional text fields (abstract_text, fulltext_text, has_abstract, etc.), but they are mostly null or unused.

**What would be needed:** (1) A text feature builder (e.g. from abstract_text) or an embedding step. (2) Config to specify which text fields to use. (3) Either extend the ColumnTransformer to include a text branch or a separate pipeline for text-derived features. (4) Data pipeline or API changes if full text is to be populated.

**Fixed-horizon outcomes:** Config and docs already distinguish proxy vs research target; no code path computes a fixed-horizon target (e.g. citations at 3 years). That would require either a different data source or a join with citation snapshots by year.

**Time-based evaluation:** Split config has `split_kind` and `time_column`; implementation for time-based split is not done. Adding it would require implementing the branch in `data/split.py` and possibly ensuring publication_date/year is available and sorted.

---

## 10. Prioritized next steps

**Short-term (transparency and correctness)**  
1. Align run metadata with actual data: e.g. in `run_pipeline_from_configs`, set `dataset_id` (and optionally processed path) from the data config used for fetch, and pass that into `build_run_metadata` so artifacts cannot claim the wrong dataset.  
2. Document that when running train/evaluate alone, `dataset_id` and `processed_path` in the experiment config define what run metadata says and what file is read.  
3. Investigate pilot corpus: why minimum citations are so high; adjust OpenAlex query or sampling if the goal is a representative benchmark.  
4. Fix or document venue_name: determine why it is 100% missing in pilot (normalization vs API) and fix or document.

**Medium-term (evaluation and target)**  
5. Implement time-based split and use it for citation experiments where appropriate.  
6. When possible, add a fixed-horizon target and keep proxy for development only.  
7. Optionally add a small script or CI step that runs diagnostics and fails if e.g. dataset_id in metrics does not match the processed file.

**Do not prioritize yet**  
- Large-scale infrastructure, MLflow, cloud, or distributed training.  
- Full-text or embedding pipelines until the metadata benchmark and evaluation design are stable.
