"""
Temporal H2 benchmark review: population summary, baseline vs XGBoost, subgroups, feature importance.

Run from repo root. Requires processed parquet at path in temporal H2 config.
If artifacts/models/ contain baseline_temporal_h2.joblib and xgb_temporal_h2.joblib, uses them;
otherwise trains and saves (same as make run-temporal-h2 for the two experiments).

Figures (target distribution, actual vs predicted) require matplotlib. Install with:
  uv sync --extra dev

Writes:
  - docs/temporal_h2_review.md
  - docs/temporal_h2_review.json
  - artifacts/diagnostics/temporal_h2_subgroup_metrics.json
  - artifacts/diagnostics/temporal_h2_feature_importance.json
  - artifacts/figures/temporal_h2_target_dist.pdf (if matplotlib installed)
  - artifacts/figures/temporal_h2_actual_vs_pred.pdf (if matplotlib installed)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scholarly_outcome_prediction.data import train_test_split_df
from scholarly_outcome_prediction.features import build_feature_matrix, build_preprocessor
from scholarly_outcome_prediction.features.targets import prepare_df_for_target
from scholarly_outcome_prediction.evaluation.metrics import compute_metrics
from scholarly_outcome_prediction.evaluation.report import load_model_pipeline
from scholarly_outcome_prediction.settings import load_experiment_config
from scholarly_outcome_prediction.utils.io import load_json, read_parquet, save_json
from scholarly_outcome_prediction.utils.seeds import set_global_seed


def _target_distribution_summary(ser: pd.Series) -> dict[str, float]:
    valid = ser.dropna()
    if len(valid) == 0:
        return {"count": 0, "min": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None, "mean": None}
    return {
        "count": int(valid.count()),
        "min": float(valid.min()),
        "q05": float(valid.quantile(0.05)),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.quantile(0.50)),
        "q75": float(valid.quantile(0.75)),
        "q95": float(valid.quantile(0.95)),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


def _zero_rate(ser: pd.Series) -> float:
    valid = ser.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid == 0).sum() / len(valid))


def run_review(
    baseline_config_path: Path | None = None,
    xgb_config_path: Path | None = None,
    processed_path: Path | None = None,
    artifacts_root: Path | None = None,
    write_plots: bool = True,
) -> dict:
    baseline_config_path = baseline_config_path or ROOT / "configs" / "experiments" / "baseline_temporal_h2.yaml"
    xgb_config_path = xgb_config_path or ROOT / "configs" / "experiments" / "xgb_temporal_h2.yaml"
    artifacts_root = artifacts_root or ROOT / "artifacts"
    base_cfg = load_experiment_config(baseline_config_path)
    xgb_cfg = load_experiment_config(xgb_config_path)
    processed_path = processed_path or ROOT / base_cfg.data.processed_path

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}. Run fetch + prepare first.")

    set_global_seed(base_cfg.split.random_state)
    full_df = read_parquet(processed_path)
    num_feat = base_cfg.features.numeric
    cat_feat = base_cfg.features.categorical
    target_name = base_cfg.target.name

    full_df, eligibility_info = prepare_df_for_target(
        full_df,
        target_name=target_name,
        target_mode=base_cfg.target.target_mode,
        horizon_years=getattr(base_cfg.target, "horizon_years", None),
        include_publication_year=getattr(base_cfg.target, "include_publication_year", True),
    )

    X, y = build_feature_matrix(
        full_df,
        numeric_features=num_feat,
        categorical_features=cat_feat,
        target_name=target_name,
        target_transform=base_cfg.target.transform or None,
    )
    full = pd.concat([X, y], axis=1)
    full = full.dropna(subset=[target_name])
    train_df, test_df = train_test_split_df(
        full,
        test_size=base_cfg.split.test_size,
        random_state=base_cfg.split.random_state,
        split_kind=getattr(base_cfg.split, "split_kind", "random"),
        time_column=getattr(base_cfg.split, "time_column", None),
        train_year_end=getattr(base_cfg.split, "train_year_end", None),
        test_year_start=getattr(base_cfg.split, "test_year_start", None),
    )

    X_train = train_df[num_feat + cat_feat]
    y_train = train_df[target_name]
    X_test = test_df[num_feat + cat_feat]
    y_test = test_df[target_name]
    train_years = train_df["publication_year"] if "publication_year" in train_df.columns else None
    test_years = test_df["publication_year"] if "publication_year" in test_df.columns else None
    # Zero rate: log1p(0)=0 so (y==0) in transformed space means zero raw target
    zero_rate_train = _zero_rate(y_train)
    zero_rate_test = _zero_rate(y_test)

    # Load or train models
    model_dir = artifacts_root / "models"
    base_pipe_path = model_dir / f"{base_cfg.experiment_name}.joblib"
    xgb_pipe_path = model_dir / f"{xgb_cfg.experiment_name}.joblib"
    if base_pipe_path.exists() and xgb_pipe_path.exists():
        base_pipe = load_model_pipeline(base_pipe_path)
        xgb_pipe = load_model_pipeline(xgb_pipe_path)
    else:
        from sklearn.pipeline import Pipeline
        from scholarly_outcome_prediction.models import get_model_builder
        model_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = build_preprocessor(num_feat, cat_feat)
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", get_model_builder(base_cfg.model.name)(params=base_cfg.model.params)),
        ])
        base_pipe.fit(X_train, y_train)
        xgb_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", get_model_builder(xgb_cfg.model.name)(params=xgb_cfg.model.params)),
        ])
        xgb_pipe.fit(X_train, y_train)
        import joblib
        joblib.dump(base_pipe, base_pipe_path)
        joblib.dump(xgb_pipe, xgb_pipe_path)

    y_pred_baseline = base_pipe.predict(X_test)
    y_pred_xgb = xgb_pipe.predict(X_test)

    # Metrics
    metrics_baseline = compute_metrics(y_test.values, y_pred_baseline, metric_names=["rmse", "mae", "r2"])
    metrics_xgb = compute_metrics(y_test.values, y_pred_xgb, metric_names=["rmse", "mae", "r2"])

    def _spearman_if_defined(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
        if len(y_true) < 2 or np.std(y_pred) < 1e-12:
            return None  # constant predictor => correlation undefined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. ConstantInputWarning when pred is near-constant
            r = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        return None if pd.isna(r) else float(r)

    spearman_baseline = _spearman_if_defined(y_test.values, y_pred_baseline)
    spearman_xgb = _spearman_if_defined(y_test.values, y_pred_xgb)

    # Population summary
    population = {
        "total_processed_rows": eligibility_info.get("n_rows_raw"),
        "target_eligible_rows": eligibility_info.get("n_eligible", len(full_df)),
        "excluded_incomplete_horizon": eligibility_info.get("n_excluded_horizon_incomplete"),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_year_range": [int(train_years.min()), int(train_years.max())] if train_years is not None else None,
        "test_year_range": [int(test_years.min()), int(test_years.max())] if test_years is not None else None,
        "train_year_counts": train_years.value_counts().sort_index().astype(int).to_dict() if train_years is not None else None,
        "test_year_counts": test_years.value_counts().sort_index().astype(int).to_dict() if test_years is not None else None,
        "target_zero_rate_train": round(zero_rate_train, 4),
        "target_zero_rate_test": round(zero_rate_test, 4),
        "target_distribution_train": _target_distribution_summary(y_train),
        "target_distribution_test": _target_distribution_summary(y_test),
    }

    # Subgroup analysis on test set
    test_full = test_df.copy()
    test_full["y_true"] = y_test.values
    test_full["y_pred_baseline"] = y_pred_baseline
    test_full["y_pred_xgb"] = y_pred_xgb
    test_full["target_zero"] = (test_full[target_name] == 0)

    def slice_metrics(g: pd.DataFrame) -> dict:
        if len(g) == 0:
            return {"n": 0, "mae_baseline": None, "mae_xgb": None, "rmse_xgb": None, "mean_target": None}
        return {
            "n": int(len(g)),
            "mean_target": float(g["y_true"].mean()),
            "zero_rate": round(float(g["target_zero"].mean()), 4),
            "mae_baseline": float(np.abs(g["y_true"].values - g["y_pred_baseline"].values).mean()),
            "mae_xgb": float(np.abs(g["y_true"].values - g["y_pred_xgb"].values).mean()),
            "rmse_xgb": float(np.sqrt(((g["y_true"].values - g["y_pred_xgb"].values) ** 2).mean())),
        }

    subgroups = {}
    if "publication_year" in test_full.columns:
        by_year = {int(yr): slice_metrics(test_full[test_full["publication_year"] == yr]) for yr in test_full["publication_year"].dropna().unique()}
        subgroups["by_publication_year"] = by_year
    subgroups["zero_target"] = slice_metrics(test_full[test_full["target_zero"]])
    subgroups["nonzero_target"] = slice_metrics(test_full[~test_full["target_zero"]])

    for col, bin_edges in [
        ("authors_count", [0, 1, 2, 5, 10, 100, np.inf]),
        ("referenced_works_count", [0, 10, 20, 50, 100, 1000, np.inf]),
    ]:
        if col not in test_full.columns:
            continue
        ser = test_full[col].astype(float)
        n_bins = len(bin_edges) - 1
        labels = [f"bin_{i}" for i in range(n_bins)]
        test_full[f"{col}_bucket"] = pd.cut(ser, bins=bin_edges, labels=labels, right=False)
        by_bucket = {}
        for b in test_full[f"{col}_bucket"].dropna().unique():
            by_bucket[str(b)] = slice_metrics(test_full[test_full[f"{col}_bucket"] == b])
        subgroups[f"by_{col}_bucket"] = by_bucket

    if "primary_topic" in test_full.columns:
        top_topics = test_full["primary_topic"].value_counts().head(8).index.tolist()
        by_topic = {}
        for t in top_topics:
            by_topic[str(t)] = slice_metrics(test_full[test_full["primary_topic"] == t])
        by_topic["other"] = slice_metrics(test_full[~test_full["primary_topic"].isin(top_topics)])
        subgroups["by_primary_topic"] = by_topic

    # Feature importance (XGBoost)
    # Build feature names manually: ColumnTransformer's categorical branch uses a Pipeline with
    # FunctionTransformer "impute" which does not provide get_feature_names_out().
    preprocessor = xgb_pipe.named_steps["preprocessor"]
    model = xgb_pipe.named_steps["model"]
    try:
        ct = preprocessor
        feat_parts = []
        if "num" in ct.named_transformers_:
            feat_parts.extend(num_feat)
        if "cat" in ct.named_transformers_:
            cat_pipe = ct.named_transformers_["cat"]
            if hasattr(cat_pipe, "named_steps") and "onehot" in cat_pipe.named_steps:
                ohe = cat_pipe.named_steps["onehot"]
                if hasattr(ohe, "get_feature_names_out"):
                    cat_names = ohe.get_feature_names_out(cat_feat) if cat_feat else ohe.get_feature_names_out()
                    feat_parts.extend(list(cat_names))
                elif hasattr(ohe, "categories_"):
                    n_cat = sum(len(c) for c in ohe.categories_)
                    feat_parts.extend([f"cat_{i}" for i in range(n_cat)])
                else:
                    feat_parts.extend([f"cat_{i}" for i in range(len(model.feature_importances_) - len(feat_parts))])
            else:
                feat_parts.extend([f"cat_{i}" for i in range(len(model.feature_importances_) - len(feat_parts))])
        feat_names = np.asarray(feat_parts)
        imp = model.feature_importances_
        if len(feat_names) != len(imp):
            feat_names = np.array([f"f{i}" for i in range(len(imp))])
        importance_list = [{"feature": str(n), "importance": float(i)} for n, i in zip(feat_names, imp)]
        importance_list.sort(key=lambda x: -x["importance"])
        feature_importance = {"feature_importances": importance_list[:30], "note": "XGBoost gain-based importance"}
    except Exception as e:
        feature_importance = {"error": str(e), "feature_importances": []}

    # Zero-inflation
    n_zero_test = int(test_full["target_zero"].sum())
    zero_inflation = {
        "test_rows_zero_target": n_zero_test,
        "test_rows_nonzero_target": int(len(test_full) - n_zero_test),
        "mae_baseline_zero": float(np.abs(test_full.loc[test_full["target_zero"], "y_true"].values - test_full.loc[test_full["target_zero"], "y_pred_baseline"].values).mean()) if n_zero_test else None,
        "mae_baseline_nonzero": float(np.abs(test_full.loc[~test_full["target_zero"], "y_true"].values - test_full.loc[~test_full["target_zero"], "y_pred_baseline"].values).mean()) if (len(test_full) - n_zero_test) else None,
        "mae_xgb_zero": float(np.abs(test_full.loc[test_full["target_zero"], "y_true"].values - test_full.loc[test_full["target_zero"], "y_pred_xgb"].values).mean()) if n_zero_test else None,
        "mae_xgb_nonzero": float(np.abs(test_full.loc[~test_full["target_zero"], "y_true"].values - test_full.loc[~test_full["target_zero"], "y_pred_xgb"].values).mean()) if (len(test_full) - n_zero_test) else None,
    }

    # Compare to other benchmarks (if metrics exist)
    metrics_dir = artifacts_root / "metrics"
    other_metrics = {}
    for name in ["baseline_representative", "xgb_representative", "baseline_temporal", "xgb_temporal", "baseline_representative_h2", "xgb_representative_h2"]:
        p = metrics_dir / f"{name}.json"
        if p.exists():
            try:
                other_metrics[name] = {"r2": load_json(p).get("r2"), "rmse": load_json(p).get("rmse"), "mae": load_json(p).get("mae")}
            except Exception:
                pass

    review = {
        "benchmark_definition": {
            "target_semantics": "Citations from publication_year through publication_year+2 (3 calendar years), from counts_by_year",
            "split_semantics": "Time-based: train publication_year <= 2018, test publication_year >= 2019",
            "eligibility_rule": "Row eligible iff publication_year + horizon_years <= max_available_citation_year",
            "feature_set_numeric": num_feat,
            "feature_set_categorical": cat_feat,
            "models_compared": ["baseline", "xgboost"],
            "limitations": [
                "Calendar-horizon is calendar-year granularity, not exact month-based window",
                "Empty/missing counts_by_year treated as zero yearly-count series",
                "Target-level vs dataset-level cited_by_count are different concepts",
            ],
        },
        "population_summary": population,
        "metrics": {
            "baseline": {**metrics_baseline, "spearman": spearman_baseline},
            "xgboost": {**metrics_xgb, "spearman": spearman_xgb},
        },
        "subgroup_metrics": subgroups,
        "zero_inflation": zero_inflation,
        "feature_importance": feature_importance,
        "other_benchmarks": other_metrics,
    }

    # Write artifacts
    docs = ROOT / "docs"
    diag = artifacts_root / "diagnostics"
    figures = artifacts_root / "figures"
    docs.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    save_json(review, docs / "temporal_h2_review.json")
    save_json(subgroups, diag / "temporal_h2_subgroup_metrics.json")
    save_json(feature_importance, diag / "temporal_h2_feature_importance.json")

    # Markdown report
    md = _render_review_md(review, population, metrics_baseline, metrics_xgb, spearman_baseline, spearman_xgb, zero_inflation, feature_importance, other_metrics)
    (docs / "temporal_h2_review.md").write_text(md, encoding="utf-8")

    if write_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Figure 1: target distribution (train vs test)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.hist(y_train, bins=30, alpha=0.7, label="Train", density=True)
            ax1.hist(y_test, bins=30, alpha=0.7, label="Test", density=True)
            ax1.set_xlabel("Target (transformed)")
            ax1.set_ylabel("Density")
            ax1.set_title("Target distribution: train vs test")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(figures / "temporal_h2_target_dist.pdf")
            plt.close(fig1)
            # Figure 2: actual vs predicted (test set)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(y_test, y_pred_xgb, alpha=0.5, s=10)
            y_max = float(np.nanmax(y_test.values)) if len(y_test) else 1.0
            ax2.plot([0, y_max], [0, y_max], "k--", label="y=x")
            ax2.set_xlabel("Actual (transformed)")
            ax2.set_ylabel("Predicted (XGBoost)")
            ax2.set_title("Actual vs predicted (test)")
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(figures / "temporal_h2_actual_vs_pred.pdf")
            plt.close(fig2)
            print("Figures written to artifacts/figures/temporal_h2_target_dist.pdf and temporal_h2_actual_vs_pred.pdf", file=sys.stderr)
        except ImportError as e:
            review["plot_error"] = str(e)
            print("Figures not generated (matplotlib required). Install with: uv sync --extra dev", file=sys.stderr)
        except Exception as e:
            review["plot_error"] = str(e)
            print(f"Figures not generated: {e}", file=sys.stderr)

    return review


def _render_review_md(review: dict, population: dict, metrics_baseline: dict, metrics_xgb: dict, spearman_baseline, spearman_xgb, zero_inflation: dict, feature_importance: dict, other_metrics: dict) -> str:
    lines = [
        "# Temporal H2 benchmark review",
        "",
        "## 1. Benchmark definition",
        "",
        "- **Target:** Citations from publication year through publication year + 2 (3 calendar years), from `counts_by_year`. Transform: `log1p`.",
        "- **Split:** Time-based. Train: `publication_year <= 2018`; test: `publication_year >= 2019`.",
        "- **Eligibility:** Row included only if `publication_year + horizon_years <= max_available_citation_year`.",
        "- **Features:** " + ", ".join(review["benchmark_definition"]["feature_set_numeric"]) + "; categorical: " + ", ".join(review["benchmark_definition"]["feature_set_categorical"]) + ".",
        "- **Models:** Baseline (median-predictor) and XGBoost.",
        "- **Limitations:** Calendar-year granularity; empty/missing `counts_by_year` → zero; target is not dataset-level `cited_by_count`.",
        "",
        "## 2. Benchmark population",
        "",
        f"- Processed rows: {population.get('total_processed_rows', 'N/A')}",
        f"- Target-eligible rows: {population.get('target_eligible_rows', 'N/A')}",
        f"- Excluded (incomplete horizon): {population.get('excluded_incomplete_horizon', 'N/A')}",
        f"- Train rows: {population.get('train_rows')}; test rows: {population.get('test_rows')}",
        f"- Train year range: {population.get('train_year_range')}; test year range: {population.get('test_year_range')}",
        f"- Target zero-rate train: {population.get('target_zero_rate_train')}; test: {population.get('target_zero_rate_test')}",
        "",
        "**Train target distribution (transformed):** " + json.dumps(population.get("target_distribution_train", {}), indent=2).replace("\n", "\n  "),
        "",
        "**Test target distribution (transformed):** " + json.dumps(population.get("target_distribution_test", {}), indent=2).replace("\n", "\n  "),
        "",
        "## 3. Baseline vs XGBoost",
        "",
        "| Metric | Baseline | XGBoost |",
        "|--------|----------|---------|",
        f"| RMSE   | {metrics_baseline.get('rmse', 0):.4f} | {metrics_xgb.get('rmse', 0):.4f} |",
        f"| MAE    | {metrics_baseline.get('mae', 0):.4f} | {metrics_xgb.get('mae', 0):.4f} |",
        f"| R²     | {metrics_baseline.get('r2', 0):.4f} | {metrics_xgb.get('r2', 0):.4f} |",
        f"| Spearman | {spearman_baseline if spearman_baseline is not None else 'N/A'} | {spearman_xgb if spearman_xgb is not None else 'N/A'} |",
        "",
        "## 4. Subgroup analysis (test set)",
        "",
        "See `artifacts/diagnostics/temporal_h2_subgroup_metrics.json` for full slices.",
        "",
    ]
    # Add short subgroup table: by_publication_year and zero vs nonzero
    subgroups = review.get("subgroup_metrics", {})
    if "by_publication_year" in subgroups:
        lines.append("| Publication year | n | MAE (XGBoost) |")
        lines.append("|------------------|---|---------------|")
        for yr, s in sorted(subgroups["by_publication_year"].items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            mae = s.get("mae_xgb")
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"
            lines.append(f"| {yr} | {s.get('n', 0)} | {mae_str} |")
        lines.append("")
    if "zero_target" in subgroups and "nonzero_target" in subgroups:
        z, nz = subgroups["zero_target"], subgroups["nonzero_target"]
        lines.append("| Slice | n | MAE baseline | MAE XGBoost |")
        lines.append("|-------|---|--------------|-------------|")
        lines.append(f"| Zero target | {z.get('n')} | {z.get('mae_baseline') or 'N/A'} | {z.get('mae_xgb') or 'N/A'} |")
        lines.append(f"| Nonzero target | {nz.get('n')} | {nz.get('mae_baseline') or 'N/A'} | {nz.get('mae_xgb') or 'N/A'} |")
        lines.append("")
    lines.extend([
        "## 5. Zero-inflation",
        "",
        f"- Test rows with target = 0: {zero_inflation.get('test_rows_zero_target')}",
        f"- Test rows with target > 0: {zero_inflation.get('test_rows_nonzero_target')}",
        f"- MAE baseline on zero-target: {zero_inflation.get('mae_baseline_zero')}; on nonzero: {zero_inflation.get('mae_baseline_nonzero')}",
        f"- MAE XGBoost on zero-target: {zero_inflation.get('mae_xgb_zero')}; on nonzero: {zero_inflation.get('mae_xgb_nonzero')}",
        "",
        "## 6. Feature signal",
        "",
    ])
    imp_list = feature_importance.get("feature_importances", [])[:15]
    if imp_list:
        lines.append("Top feature importances (XGBoost):")
        lines.append("")
        for item in imp_list:
            lines.append(f"- {item.get('feature', '')}: {item.get('importance', 0):.4f}")
        lines.append("")
        lines.append("Full list: `artifacts/diagnostics/temporal_h2_feature_importance.json`.")
    else:
        lines.append("See `artifacts/diagnostics/temporal_h2_feature_importance.json`. " + (feature_importance.get("error") or ""))
    lines.append("")
    lines.extend([
        "## 7. Comparison to other benchmarks",
        "",
    ])
    if other_metrics:
        for name, m in other_metrics.items():
            r2, rmse, mae = m.get("r2"), m.get("rmse"), m.get("mae")
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            lines.append(f"- **{name}**: R²={r2_str}, RMSE={rmse}, MAE={mae}")
        lines.append("")
    else:
        lines.append("No other benchmark metrics found in artifacts/metrics.")
        lines.append("")
    lines.extend([
        "## 8. Interpretation and summary",
        "",
        "- **What the temporal H2 result suggests:** Generalization to later publication years with a fixed 3-year citation window; R² and Spearman indicate whether metadata adds predictive signal beyond the baseline.",
        "- **Is the model learning real signal?** Compare XGBoost R²/Spearman to baseline; check feature importance and subgroup MAE to see if gains are broad or driven by a few slices.",
        "- **Benchmark credibility:** If XGBoost materially outperforms baseline and error is stable across years and zero/nonzero slices, the benchmark is usable for further work.",
        "- **Main limitations:** Calendar-year target approximation; metadata-only; zero-inflation; simple baselines.",
        "- **Sensible next steps:** Stronger baselines (e.g. year-stratified median), feature engineering (venue/topic embeddings or bins), calibration for zero-heavy targets, or (later) text features.",
        "",
        "## 9. Limitations (explicit)",
        "",
        "- Calendar-horizon target is an approximation (calendar years, not exact months).",
        "- Empty/missing `counts_by_year` is treated as zero yearly-count series.",
        "- Target distribution is not the same as dataset-level `cited_by_count`.",
        "- Venue/topic semantics may reflect source data rather than canonical taxonomy.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    run_review()
