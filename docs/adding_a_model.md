# Adding a new model to the benchmark suite

This guide explains how to add a new model so it is trained, evaluated, and **correctly classified** in benchmark comparison and related reports. **Benchmark metadata is required** for current train/evaluate jobs; omission causes the CLI to fail with a clear migration hint.

---

## 1. Implement and register the model

- **Implement** your estimator under `src/scholarly_outcome_prediction/models/`. It should be scikit-learn compatible (`fit`, `predict`).
- **Register** it in `src/scholarly_outcome_prediction/models/registry.py`: add a builder function and register it in the `registry` dict with a `model.name` (e.g. `"my_model"`).

See existing examples: `ridge_model.py`, `hurdle_baseline.py`, `year_conditioned_baseline.py`.

---

## 2. Add experiment config(s)

Create one or more experiment configs under `configs/experiments/` (and optionally `configs/experiments/ablations/` if this model is used in ablations).

Each config must specify:

- `experiment_name`, `data`, `target`, `features`, `split`, `model` (name + params), `evaluation` — same as any existing experiment.
- **Benchmark metadata (required for train/evaluate):**
  - Add a **`benchmark:`** block with:
    - **`benchmark_mode`**: one of `representative_proxy`, `temporal_proxy`, `representative_h2`, `temporal_h2`.
    - **`model_family`**: e.g. `trivial_baseline`, `linear_baseline`, `tree_model`, `hurdle_baseline`, `diagnostic_baseline`, `count_aware_glm`, or a new family label you introduce.
    - **`is_diagnostic_model`**: `true` only if this model is for interpretation/diagnostic use (e.g. year-conditioned baseline), not a primary comparator. Default `false`.

Example (primary model):

```yaml
experiment_name: my_model_temporal_h2
task_type: regression
benchmark:
  benchmark_mode: temporal_h2
  model_family: tree_model   # or a new label, e.g. custom_ensemble
  is_diagnostic_model: false

data:
  processed_path: "data/processed/openalex_temporal_articles_1000.parquet"
  dataset_id: openalex_temporal_articles_1000
# ... target, features, split, model, evaluation
```

Example (diagnostic model):

```yaml
benchmark:
  benchmark_mode: temporal_h2
  model_family: diagnostic_baseline
  is_diagnostic_model: true
```

These fields are written into the **metrics JSON** at evaluate time. Benchmark-analysis uses them as the **authoritative** classification. When reading **older metrics artifacts** that lack these fields, benchmark-analysis still infers from `experiment_name` / `dataset_id` and labels the source as `legacy_inferred`; that path is for backward compatibility only, not for new configs.

---

## 3. Run the pipeline

- **Train:** `uv run scholarly-outcome-prediction train --config configs/experiments/my_model_temporal_h2.yaml`
- **Evaluate:** `uv run scholarly-outcome-prediction evaluate --config configs/experiments/my_model_temporal_h2.yaml`

Or use the `run` command with a data config and two experiment configs; or add a Makefile target that runs train + evaluate for your config.

---

## 4. How benchmark comparison classifies your model

- **Benchmark mode** (which row in the comparison table): from **`benchmark_mode`** in the metrics JSON (required in current configs). Older artifacts without it are classified via legacy inference and labeled `legacy_inferred`.
- **Model family** (e.g. “linear baseline”, “tree model”): from **`model_family`** in the metrics JSON (required in current configs). Older artifacts use a legacy fallback map.
- **Diagnostic vs primary**: from **`is_diagnostic_model`** in the metrics JSON (required in current configs). Older artifacts use a legacy fallback set of model names.

**Active benchmark suite:** Only models not in `BENCHMARK_EXCLUDED_MODELS` appear in the comparison table. Currently excluded: `tweedie` (retained in registry for experimental use; poor methodological fit for the shared target/preprocessing framework). Supported active benchmark models include baseline, ridge, elastic_net, extra_trees, hist_gradient_boosting, xgboost, hurdle, year_conditioned (diagnostic).

**Current configs must declare the `benchmark` block** so that train/evaluate succeed and metrics are self-describing. Inference from naming exists only for reading historical metrics artifacts.

---

## 5. Adding an ablation that uses your model

If you add an **ablation** (e.g. “XGBoost temporal H2 without feature X”):

- Add an **`ablation:`** block to the experiment config with:
  - **`name`**: short identifier (e.g. `no_publication_year`).
  - **`features_removed`**: list of feature names removed in this run. This is the **single source of truth** for ablation reports; do not duplicate it in code.
  - **`ablation_type`** (optional): `coarse` or `numeric_fine` for reporting.

Example:

```yaml
ablation:
  name: no_my_feature
  features_removed: [my_feature]
  ablation_type: numeric_fine
```

These fields are written into the metrics JSON. The ablation review uses **`ablation_features_removed`** from the artifact when present. Older artifacts without it are still readable; their classification is labeled `legacy_inferred`.

---

## 6. Summary checklist

- [ ] Model implemented and registered in `models/registry.py`.
- [ ] Experiment config(s) added with correct `data`, `target`, `features`, `split`, `model`, `evaluation`.
- [ ] **`benchmark:`** block set (benchmark_mode, model_family, is_diagnostic_model) so comparison classifies the run explicitly.
- [ ] If the run is an ablation: **`ablation:`** block set (name, features_removed, optional ablation_type).
- [ ] Run train + evaluate; run `benchmark-analysis` to regenerate comparison and ablation reports.

No need to change benchmark-analysis code when adding a new model or ablation. Add the required `benchmark` (and `ablation` when applicable) block to your YAML and run the pipeline.
