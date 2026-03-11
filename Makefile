PYTHON ?= python

.PHONY: install lint format test clean run-example run-pilot run-pilot-time run-representative-pilot run-temporal-pilot run-representative-h2 run-temporal-h2 run-temporal-h2-baselines run-temporal-h2-hurdle run-temporal-h2-ablations run-new-benchmark-models run-representative-pilot-ridge run-temporal-pilot-ridge run-representative-h2-ridge run-full-benchmark validate-representative-pilot validate-temporal-pilot validate-latest-pilot profile-representative-pilot profile-temporal-pilot temporal-h2-review benchmark-analysis

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

# Remove transient data, artifacts, and cache (keeps .gitkeep in data/ and artifacts/).
clean:
	@for dir in data/raw data/interim data/processed artifacts/models artifacts/metrics artifacts/figures artifacts/reports artifacts/diagnostics artifacts/debug; do \
		if [ -d "$$dir" ]; then find "$$dir" -depth -mindepth 1 ! -name .gitkeep -exec rm -rf {} \; 2>/dev/null || true; fi; \
	done
	@rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true
	@find src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run-example:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_sample_100.yaml \
		--baseline-config configs/experiments/baseline_regression.yaml \
		--xgb-config configs/experiments/xgb_regression.yaml

run-pilot:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_pilot.yaml \
		--baseline-config configs/experiments/baseline_regression.yaml \
		--xgb-config configs/experiments/xgb_regression.yaml

# New pilot: journal+proceedings articles only, sort=publication_date:asc; time-based train/test split.
run-pilot-time:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_pilot_articles.yaml \
		--baseline-config configs/experiments/baseline_regression_time.yaml \
		--xgb-config configs/experiments/xgb_regression_time.yaml

# Representative pilot: article-only, stratify_by_year for multi-year sample; random split for benchmarking.
run-representative-pilot:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_representative_articles_1000.yaml \
		--baseline-config configs/experiments/baseline_representative.yaml \
		--xgb-config configs/experiments/xgb_representative.yaml

# Temporal pilot: article-only, stratify_by_year for multi-year sample; train 2015-2018, test 2019-2020.
run-temporal-pilot:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_temporal_articles_1000.yaml \
		--baseline-config configs/experiments/baseline_temporal.yaml \
		--xgb-config configs/experiments/xgb_temporal.yaml

# Representative pilot with 2-year calendar-horizon target (counts_by_year; requires fresh fetch with counts_by_year).
run-representative-h2:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_representative_articles_1000.yaml \
		--baseline-config configs/experiments/baseline_representative_h2.yaml \
		--xgb-config configs/experiments/xgb_representative_h2.yaml

# Temporal pilot with 2-year calendar-horizon target.
run-temporal-h2:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_temporal_articles_1000.yaml \
		--baseline-config configs/experiments/baseline_temporal_h2.yaml \
		--xgb-config configs/experiments/xgb_temporal_h2.yaml

# Temporal H2 with Ridge + year-conditioned baselines (run after run-temporal-h2 for full comparison).
run-temporal-h2-baselines:
	uv run scholarly-outcome-prediction run \
		--data-config configs/data/openalex_temporal_articles_1000.yaml \
		--baseline-config configs/experiments/ridge_temporal_h2.yaml \
		--xgb-config configs/experiments/year_conditioned_temporal_h2.yaml

# Temporal H2 with hurdle baseline (train + evaluate). Requires data from make run-temporal-h2.
run-temporal-h2-hurdle:
	uv run scholarly-outcome-prediction train --config configs/experiments/hurdle_temporal_h2.yaml
	uv run scholarly-outcome-prediction evaluate --config configs/experiments/hurdle_temporal_h2.yaml

# Run ablation experiments (train + evaluate each). Requires data from make run-temporal-h2.
# Coarse + fine-grained numeric (no_referenced_works_count, no_authors_count, no_institutions_count).
run-temporal-h2-ablations:
	@echo "Running ablation experiments (data must exist from make run-temporal-h2)..."
	@for c in configs/experiments/ablations/xgb_temporal_h2_no_publication_year.yaml \
		configs/experiments/ablations/xgb_temporal_h2_no_venue_name.yaml \
		configs/experiments/ablations/xgb_temporal_h2_no_primary_topic.yaml \
		configs/experiments/ablations/xgb_temporal_h2_numeric_only.yaml \
		configs/experiments/ablations/xgb_temporal_h2_categorical_only.yaml \
		configs/experiments/ablations/xgb_temporal_h2_no_referenced_works_count.yaml \
		configs/experiments/ablations/xgb_temporal_h2_no_authors_count.yaml \
		configs/experiments/ablations/xgb_temporal_h2_no_institutions_count.yaml; do \
	  echo "Training and evaluating $$c..."; \
	  uv run scholarly-outcome-prediction train --config $$c && uv run scholarly-outcome-prediction evaluate --config $$c || exit 1; \
	done

# Ridge baseline for representative proxy (train + evaluate only; run run-representative-pilot first).
run-representative-pilot-ridge:
	uv run scholarly-outcome-prediction train --config configs/experiments/ridge_representative.yaml
	uv run scholarly-outcome-prediction evaluate --config configs/experiments/ridge_representative.yaml

# Ridge baseline for temporal proxy (train + evaluate only; run run-temporal-pilot first).
run-temporal-pilot-ridge:
	uv run scholarly-outcome-prediction train --config configs/experiments/ridge_temporal.yaml
	uv run scholarly-outcome-prediction evaluate --config configs/experiments/ridge_temporal.yaml

# Ridge baseline for representative H2 (train + evaluate only; run run-representative-h2 first).
run-representative-h2-ridge:
	uv run scholarly-outcome-prediction train --config configs/experiments/ridge_representative_h2.yaml
	uv run scholarly-outcome-prediction evaluate --config configs/experiments/ridge_representative_h2.yaml

# Train and evaluate new benchmark models (elastic_net, extra_trees, hist_gradient_boosting) for all four modes.
# Requires data from run-representative-pilot, run-representative-h2, run-temporal-pilot, run-temporal-h2.
run-new-benchmark-models:
	@echo "Training and evaluating new benchmark models (elastic_net, extra_trees, hist_gradient_boosting)..."
	@for c in configs/experiments/elastic_net_representative.yaml configs/experiments/elastic_net_temporal.yaml \
		configs/experiments/elastic_net_representative_h2.yaml configs/experiments/elastic_net_temporal_h2.yaml \
		configs/experiments/extra_trees_representative.yaml configs/experiments/extra_trees_temporal.yaml \
		configs/experiments/extra_trees_representative_h2.yaml configs/experiments/extra_trees_temporal_h2.yaml \
		configs/experiments/hist_gradient_boosting_representative.yaml configs/experiments/hist_gradient_boosting_temporal.yaml \
		configs/experiments/hist_gradient_boosting_representative_h2.yaml configs/experiments/hist_gradient_boosting_temporal_h2.yaml; do \
	  echo "Training and evaluating $$c..."; \
	  uv run scholarly-outcome-prediction train --config $$c && uv run scholarly-outcome-prediction evaluate --config $$c || exit 1; \
	done

# Generate unified benchmark comparison and ablation review from artifacts/metrics.
benchmark-analysis:
	uv run scholarly-outcome-prediction benchmark-analysis

# Run the full benchmark suite: all four modes (representative proxy, representative H2, temporal proxy, temporal H2)
# plus ridge for each mode, temporal H2 baselines (ridge, year_conditioned), hurdle, new models (elastic_net, extra_trees, hist_gradient_boosting), ablations; then comparison reports.
# Ridge for temporal H2 is included in run-temporal-h2-baselines; ridge for other three modes via run-*-ridge.
run-full-benchmark: run-representative-pilot run-representative-pilot-ridge run-representative-h2 run-representative-h2-ridge run-temporal-pilot run-temporal-pilot-ridge run-temporal-h2 run-temporal-h2-baselines run-temporal-h2-hurdle run-new-benchmark-models run-temporal-h2-ablations benchmark-analysis

# Validate representative pilot dataset.
validate-representative-pilot:
	uv run scholarly-outcome-prediction validate --data-config configs/data/openalex_representative_articles_1000.yaml

# Validate temporal pilot dataset.
validate-temporal-pilot:
	uv run scholarly-outcome-prediction validate --data-config configs/data/openalex_temporal_articles_1000.yaml

# Audit row-level overlap between representative and temporal processed datasets (by openalex_id).
# Run after fetch+prepare for both datasets to verify they are distinct. Output: artifacts/diagnostics/*_overlap_audit.json and .md.
audit-dataset-overlap:
	uv run scholarly-outcome-prediction audit-dataset-overlap \
		--left data/processed/openalex_representative_articles_1000.parquet \
		--right data/processed/openalex_temporal_articles_1000.parquet \
		--label-left representative \
		--label-right temporal \
		--out-dir artifacts/diagnostics

# Alias for validate-representative-pilot.
validate-latest-pilot: validate-representative-pilot

# Regenerate diagnostics for the representative pilot dataset (stamped with dataset_id and run_id).
profile-representative-pilot:
	uv run python scripts/generate_diagnostics.py --processed data/processed/openalex_representative_articles_1000.parquet --dataset-id openalex_representative_articles_1000

# Regenerate diagnostics for the temporal pilot dataset.
profile-temporal-pilot:
	uv run python scripts/generate_diagnostics.py --processed data/processed/openalex_temporal_articles_1000.parquet --dataset-id openalex_temporal_articles_1000

# Generate temporal H2 benchmark review (population, baseline vs XGBoost, subgroups, feature importance).
# Requires processed temporal data and optionally pre-trained models from make run-temporal-h2.
temporal-h2-review:
	uv run python scripts/run_temporal_h2_review.py