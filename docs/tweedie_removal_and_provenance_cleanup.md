# Tweedie removal and provenance cleanup

Summary of the narrow cleanup pass: remove Tweedie from the active benchmark suite and fix dataset provenance semantics for representative vs temporal.

---

## 1. What was removed or deactivated

### Tweedie in the active benchmark path

- **Benchmark comparison**: Tweedie is excluded from the comparison table. In `evaluation/benchmark_analysis.py`, `BENCHMARK_EXCLUDED_MODELS = {"tweedie"}` was added; any metrics row with `model_name == "tweedie"` is skipped when building `rows`. Tweedie was also removed from `expected_models` so it is no longer listed as a missing benchmark.
- **Makefile**: `run-new-benchmark-models` and `run-full-benchmark` no longer run the four Tweedie experiment configs. Comments were updated to drop Tweedie from the list of “new” benchmark models.
- **Experiment configs**: The four Tweedie YAMLs were moved from `configs/experiments/` to `configs/experiments/experimental/` and each file was given an EXPERIMENTAL comment. They are no longer part of the default benchmark run.

### What was retained

- **Implementation and registry**: `tweedie_model.py` and the `tweedie` entry in `models/registry.py` are unchanged. The model remains available for training/evaluation (e.g. using configs under `configs/experiments/experimental/`). It is clearly non-default and excluded from benchmark comparison.
- **Legacy artifact reading**: `MODEL_FAMILY_FALLBACK` still includes `"tweedie": "count_aware_glm"` so older metrics JSONs that mention Tweedie can still be read; those rows are simply filtered out of the comparison output.

---

## 2. Tweedie: deleted vs retained as experimental

**Retained as experimental.** Tweedie was not deleted from the repo. It was:

- Removed from the **active** benchmark set (comparison table, expected_models, Makefile targets, and main experiment config directory).
- Retained in the **registry** and in code; configs were moved to `configs/experiments/experimental/` with a README explaining that Tweedie is not part of the default suite and why (methodological fit with the shared target/preprocessing framework).

---

## 3. Provenance wording changes

### Authority of `dataset_mode`

Previously, `_selection_strategy_summary()` in `validation/dataset_validation.py` derived the label mainly from **acquisition parameters** (`stratify_by_year`, `use_random_sample`). So a temporal dataset that used `use_random_sample=true` (e.g. temporal pilot with within-year random sampling) could get the same summary as a representative dataset (“representative: … within-year random”), which was misleading.

Now **`dataset_mode` is authoritative** for the semantics described in the summary:

- **Representative**: The summary states that the dataset is intended to approximate a broad article sample, that random within-year sampling is acceptable, and that the split strategy may be random or representative-oriented. Acquisition details are appended for clarity.
- **Temporal**: The summary states that the dataset is intended for forward-time generalization and that the **key benchmark distinction is time-ordered evaluation** (train on past, test on future). It explicitly says: “Do not interpret this dataset as a representative sample; split semantics define the benchmark.” Acquisition is then described (cursor per year vs within-year random plus time-ordered split), so it is clear that acquisition and evaluation/split semantics are both documented.

### Before/after phrasing

**Before (temporal dataset with `stratify_by_year=true`, `use_random_sample=true`):**

- `selection_strategy_summary`: `"representative: stratify_by_year=true, use_random_sample=true (within-year random)"`  
- So a temporal validation report could read “representative” and “within-year random” and obscure the fact that the benchmark is time-ordered evaluation.

**After (same config, `dataset_mode="temporal"`):**

- `selection_strategy_summary`: `"Temporal dataset: intended for forward-time generalization. The key benchmark distinction is time-ordered evaluation (train on past, test on future). Do not interpret this dataset as a representative sample; split semantics define the benchmark. Acquisition: stratify_by_year=true, use_random_sample=true (within-year random sample per year); evaluation uses time-ordered train/test split."`  
- So the artifact clearly states that the dataset is temporal, that the distinction is time-ordered evaluation, and that acquisition may still use within-year random sampling before applying the time split.

**After (representative, same acquisition params):**

- `selection_strategy_summary`: `"Representative dataset: intended to approximate a broad article sample. Random within-year sampling is acceptable; split strategy may be random or otherwise representative-oriented. Acquisition: stratify_by_year=true, use_random_sample=true (within-year random sampling)."`  
- So representative reports clearly describe broad-sample intent and representative-oriented split, and acquisition is still explicit.

---

## 4. Other updates

- **Docs**: `docs/job_yml_schema.md` no longer lists `tweedie` in the supported benchmark model examples and notes that active benchmark models exclude experimental entries. `docs/adding_a_model.md` now states that the active benchmark suite is defined by `BENCHMARK_EXCLUDED_MODELS`, that Tweedie is excluded (and why), and lists the supported active benchmark models.
- **Tests**: Tests were updated so the active benchmark suite does not assume Tweedie: `test_new_benchmark_models_configs` no longer includes Tweedie configs or family; `test_models` asserts only the active benchmark model names; `test_benchmark_analysis` has `test_benchmark_comparison_excludes_tweedie` to assert Tweedie is omitted from the comparison. A new validation test checks that provenance wording distinguishes representative vs temporal (e.g. “Temporal dataset”, “time-ordered”, “Do not interpret this dataset as a representative sample”).

---

## 5. Backward compatibility

- No broad artifact schema change: existing JSON/MD shapes (e.g. `provenance.selection_strategy_summary`, `dataset_mode`) are unchanged; only the **text content** of the summary was made explicit and mode-aware.
- Legacy metrics that contain Tweedie are still loaded; Tweedie rows are dropped only when building the benchmark comparison. Historical artifact readers that do not depend on Tweedie being in the comparison table remain valid.
