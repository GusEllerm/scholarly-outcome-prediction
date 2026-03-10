# Adding a new model to the benchmark suite

This guide explains how to add a new model so it is trained, evaluated, and **correctly classified** in benchmark comparison and related reports without relying on naming conventions.

---

## 1. Implement and register the model

- **Implement** your estimator under `src/scholarly_outcome_prediction/models/`. It should be scikit-learn compatible (`fit`, `predict`).
- **Register** it in `src/scholarly_outcome_prediction/models/registry.py`: add a builder function and register it in the `registry` dict with a `model.name` (e.g. `"my_model"`).

See existing examples: `ridge_model.py`, `hurdle_baseline.py`, `year_conditioned_baseline.py`.

---

## 2. Add experiment config(s)

Create one or more experiment configs under `configs/experiments/` (and optionally `configs/experiments/ablations/` if this model is used in ablations).

Each config should specify:

- `experiment_name`, `data`, `target`, `features`, `split`, `model` (name + params), `evaluation` — same as any existing experiment.
- **Benchmark metadata** (so comparison reports classify the run correctly):
  - Add a **`benchmark:`** block with:
    - **`benchmark_mode`**: one of `representative_proxy`, `temporal_proxy`, `representative_h2`, `temporal_h2`.
    - **`model_family`**: e.g. `trivial_baseline`, `linear_baseline`, `tree_model`, `hurdle_baseline`, `diagnostic_baseline`, or a new family label you introduce.
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

These fields are written into the **metrics JSON** at evaluate time. Benchmark-analysis then uses them as the **authoritative** classification; it only falls back to inferring from `experiment_name` / `dataset_id` when they are missing (e.g. older artifacts).

---

## 3. Run the pipeline

- **Train:** `uv run scholarly-outcome-prediction train --config configs/experiments/my_model_temporal_h2.yaml`
- **Evaluate:** `uv run scholarly-outcome-prediction evaluate --config configs/experiments/my_model_temporal_h2.yaml`

Or use the `run` command with a data config and two experiment configs; or add a Makefile target that runs train + evaluate for your config.

---

## 4. How benchmark comparison classifies your model

- **Benchmark mode** (which row in the comparison table): from **`benchmark_mode`** in the metrics JSON when present; otherwise inferred from experiment name and dataset_id.
- **Model family** (e.g. “linear baseline”, “tree model”): from **`model_family`** in the metrics JSON when present; otherwise from a fallback map keyed by `model_name`.
- **Diagnostic vs primary**: from **`is_diagnostic_model`** in the metrics JSON when present; otherwise from a fallback set of model names. Diagnostic models are marked so reviewers do not treat them as primary baselines.

So: **declaring `benchmark` in your experiment config is what makes classification explicit and robust.** You do not need to follow a special naming scheme for the experiment or dataset.

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

These fields are written into the metrics JSON. The ablation review uses **`ablation_features_removed`** from the artifact when present; only older artifacts without it use a fallback mapping.

---

## 6. Summary checklist

- [ ] Model implemented and registered in `models/registry.py`.
- [ ] Experiment config(s) added with correct `data`, `target`, `features`, `split`, `model`, `evaluation`.
- [ ] **`benchmark:`** block set (benchmark_mode, model_family, is_diagnostic_model) so comparison classifies the run explicitly.
- [ ] If the run is an ablation: **`ablation:`** block set (name, features_removed, optional ablation_type).
- [ ] Run train + evaluate; run `benchmark-analysis` to regenerate comparison and ablation reports.

No need to change benchmark-analysis code or naming conventions when adding a new model or ablation — only add config and run the pipeline.
