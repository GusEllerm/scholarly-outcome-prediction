"""Unified benchmark comparison and ablation review from metrics artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scholarly_outcome_prediction.utils.io import load_json, save_json

# Canonical benchmark modes and target modes for comparison table
BENCHMARK_MODES = ["representative_proxy", "temporal_proxy", "representative_h2", "temporal_h2"]
ABLATION_FULL_EXPERIMENT = "xgb_temporal_h2"
# Coarse = feature-group ablations; numeric_fine = single numeric feature removed
ABLATION_FEATURES_REMOVED: dict[str, list[str]] = {
    "no_publication_year": ["publication_year"],
    "no_venue_name": ["venue_name"],
    "no_primary_topic": ["primary_topic"],
    "numeric_only": ["type", "language", "venue_name", "primary_topic", "open_access_is_oa"],
    "categorical_only": ["publication_year", "referenced_works_count", "authors_count", "institutions_count"],
    "no_referenced_works_count": ["referenced_works_count"],
    "no_authors_count": ["authors_count"],
    "no_institutions_count": ["institutions_count"],
}
ABLATION_TYPE_COARSE = {"no_publication_year", "no_venue_name", "no_primary_topic", "numeric_only", "categorical_only"}
MODEL_FAMILY: dict[str, str] = {
    "baseline": "trivial baseline",
    "ridge": "linear baseline",
    "year_conditioned": "diagnostic baseline",
    "hurdle": "hurdle baseline",
    "xgboost": "tree model",
}
DIAGNOSTIC_ONLY_MODELS = {"year_conditioned"}


def _infer_benchmark_mode(experiment_name: str, data: dict[str, Any]) -> str | None:
    """Infer benchmark mode: representative_proxy | temporal_proxy | representative_h2 | temporal_h2."""
    name = (experiment_name or "").lower()
    dataset_id = (data.get("dataset_id") or data.get("effective_dataset_id") or "").lower()
    rep = "representative" in name or "representative" in dataset_id
    temp = "temporal" in name or "temporal" in dataset_id
    h2 = "_h2" in name or "h2" in name or data.get("target_mode") == "calendar_horizon"
    if rep and h2:
        return "representative_h2"
    if rep:
        return "representative_proxy"
    if temp and h2:
        return "temporal_h2"
    if temp:
        return "temporal_proxy"
    return None


def _infer_target_mode(experiment_name: str, data: dict[str, Any]) -> str:
    """Infer target mode: proxy | h2 (calendar_horizon)."""
    if data.get("target_mode") == "calendar_horizon":
        return "h2"
    name = (experiment_name or "").lower()
    if "_h2" in name or "h2" in name:
        return "h2"
    return "proxy"


def _is_ablation(experiment_name: str) -> bool:
    """True if experiment is an ablation of xgb_temporal_h2."""
    if not experiment_name or not experiment_name.startswith("xgb_temporal_h2_"):
        return False
    suffix = experiment_name.replace("xgb_temporal_h2_", "")
    return suffix in ABLATION_FEATURES_REMOVED


def _ablation_name(experiment_name: str) -> str | None:
    """Return ablation name (e.g. no_publication_year) or None."""
    if not _is_ablation(experiment_name):
        return None
    return experiment_name.replace("xgb_temporal_h2_", "")


def load_all_metrics(artifacts_root: Path) -> list[dict[str, Any]]:
    """Load every JSON from artifacts_root/metrics."""
    metrics_dir = artifacts_root / "metrics"
    if not metrics_dir.exists():
        return []
    out = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            data = load_json(path)
            data["_metrics_file"] = path.name
            out.append(data)
        except Exception:
            continue
    return out


def build_benchmark_comparison(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Build unified comparison: rows per (benchmark_mode, model), plus missing list."""
    generated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for data in metrics_list:
        exp = data.get("experiment_name") or data.get("_metrics_file", "").replace(".json", "")
        bm = _infer_benchmark_mode(exp, data)
        if bm is None:
            continue
        model = data.get("model_name", "unknown")
        key = (bm, model)
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, Any] = {
            "benchmark_mode": bm,
            "target_mode": _infer_target_mode(exp, data),
            "split_kind": data.get("split_kind"),
            "model_name": model,
            "model_family": MODEL_FAMILY.get(model, "other"),
            "is_diagnostic_only": model in DIAGNOSTIC_ONLY_MODELS,
            "experiment_name": exp,
            "rmse": data.get("rmse"),
            "mae": data.get("mae"),
            "r2": data.get("r2"),
        }
        zi = data.get("zero_inflation") or {}
        row["test_zero_rate"] = zi.get("test_zero_rate")
        row["mae_zero_target"] = zi.get("mae_zero_target")
        row["mae_nonzero_target"] = zi.get("mae_nonzero_target")
        row["dataset_id"] = data.get("dataset_id") or data.get("effective_dataset_id")
        row["train_year_end"] = data.get("train_year_end")
        row["test_year_start"] = data.get("test_year_start")
        row["target_name"] = data.get("target_name")
        row["run_id"] = data.get("run_id")
        rows.append(row)

    # Missing benchmarks: expected (benchmark_mode, model) combinations
    expected_models = ["baseline", "ridge", "year_conditioned", "xgboost", "hurdle"]
    found_modes = {r["benchmark_mode"] for r in rows}
    missing: list[dict[str, Any]] = []
    for bm in BENCHMARK_MODES:
        for model in expected_models:
            if (bm, model) in seen:
                continue
            missing.append({
                "benchmark_mode": bm,
                "model_name": model,
                "note": "No metrics artifact found for this combination.",
            })

    interpretation_notes: dict[str, str] = {
        "ridge": "Ridge is a standard comparator across all four benchmark modes; compare to XGBoost to see if linear baseline remains competitive.",
        "year_conditioned": "Year-conditioned baseline is a diagnostic only: under temporal split, test years are unseen in training so it predicts global median for all test rows; do not interpret as a strong baseline failure.",
        "ablation_numeric": "Fine-grained numeric ablations (no_referenced_works_count, no_authors_count, no_institutions_count) show which single numeric features carry most signal; see ablation_review.",
    }
    return {
        "report_scope": "benchmark_comparison",
        "report_name": "Unified benchmark comparison",
        "generated_at": generated_at,
        "benchmark_modes_compared": BENCHMARK_MODES,
        "rows": rows,
        "missing": missing,
        "interpretation_notes": interpretation_notes,
        "notes": "Eligibility filtering and target semantics (e.g. horizon) apply per run; see run metadata in each metrics JSON. Models with is_diagnostic_only=true are for interpretation only, not as primary baselines.",
    }


def build_ablation_review(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Build ablation comparison: each ablation vs full xgb_temporal_h2, with deltas."""
    generated_at = datetime.now(timezone.utc).isoformat()
    full_metrics: dict[str, Any] | None = None
    for data in metrics_list:
        if (data.get("experiment_name") or "").strip() == ABLATION_FULL_EXPERIMENT:
            full_metrics = data
            break
    ablations: list[dict[str, Any]] = []
    for data in metrics_list:
        exp = data.get("experiment_name") or ""
        name = _ablation_name(exp)
        if name is None:
            continue
        features_removed = ABLATION_FEATURES_REMOVED.get(name, [])
        row: dict[str, Any] = {
            "ablation_name": name,
            "experiment_name": exp,
            "features_removed": features_removed,
            "ablation_type": "coarse" if name in ABLATION_TYPE_COARSE else "numeric_fine",
            "benchmark_mode": "temporal_h2",
            "rmse": data.get("rmse"),
            "mae": data.get("mae"),
            "r2": data.get("r2"),
        }
        zi = data.get("zero_inflation") or {}
        row["test_zero_rate"] = zi.get("test_zero_rate")
        row["mae_zero_target"] = zi.get("mae_zero_target")
        row["mae_nonzero_target"] = zi.get("mae_nonzero_target")
        if full_metrics is not None:
            row["delta_rmse"] = (data.get("rmse") - full_metrics.get("rmse")) if data.get("rmse") is not None and full_metrics.get("rmse") is not None else None
            row["delta_mae"] = (data.get("mae") - full_metrics.get("mae")) if data.get("mae") is not None and full_metrics.get("mae") is not None else None
            row["delta_r2"] = (data.get("r2") - full_metrics.get("r2")) if data.get("r2") is not None and full_metrics.get("r2") is not None else None
        else:
            row["delta_rmse"] = row["delta_mae"] = row["delta_r2"] = None
        row["interpretation"] = _ablation_interpretation(name, row)
        row["interpretation_tag"] = _ablation_interpretation_tag(row)
        ablations.append(row)

    return {
        "report_scope": "ablation_review",
        "report_name": "Metadata ablation comparison",
        "generated_at": generated_at,
        "full_model_experiment": ABLATION_FULL_EXPERIMENT,
        "full_model_available": full_metrics is not None,
        "ablations": ablations,
        "empty_hint": None if ablations else (
            "No ablation runs found. To populate: run train and evaluate for each config in "
            "configs/experiments/ablations/ (coarse and fine-grained numeric: no_referenced_works_count, "
            "no_authors_count, no_institutions_count), or use: make run-temporal-h2-ablations (after make run-temporal-h2)."
        ),
    }


def _ablation_interpretation(ablation_name: str, row: dict[str, Any]) -> str:
    """Short interpretation of what the ablation suggests."""
    delta_r2 = row.get("delta_r2")
    delta_mae = row.get("delta_mae")
    if delta_r2 is not None:
        if delta_r2 < -0.05:
            return f"Removing {ablation_name} hurts R² substantially; this feature group carries signal."
        if delta_r2 > 0.02:
            return f"Removing {ablation_name} slightly improves R²; possible overfitting or noise in this group."
    if delta_mae is not None and delta_mae > 0.1:
        return f"Removing {ablation_name} increases MAE; feature group contributes to accuracy."
    return f"Ablation {ablation_name}: see metric deltas vs full model."


def _ablation_interpretation_tag(row: dict[str, Any]) -> str:
    """Tag for quick scan: high / moderate / low / negligible impact."""
    delta_r2 = row.get("delta_r2")
    delta_mae = row.get("delta_mae")
    if delta_r2 is not None:
        if delta_r2 < -0.10:
            return "high impact"
        if delta_r2 < -0.03:
            return "moderate impact"
        if delta_r2 > 0.02:
            return "negligible / possibly noisy"
    if delta_mae is not None and delta_mae > 0.05:
        return "moderate impact"
    if delta_mae is not None and delta_mae > 0.02:
        return "low impact"
    return "low impact"


def comparison_to_md(payload: dict[str, Any]) -> str:
    """Render benchmark_comparison payload as Markdown."""
    lines = [
        "# " + payload.get("report_name", "Benchmark comparison"),
        "",
        f"**Generated:** {payload.get('generated_at', '')}",
        "",
        "## Rows (by benchmark mode and model)",
        "",
        "| Benchmark mode | Target | Model | Family | Diagnostic? | RMSE | MAE | R² | Zero rate | MAE (zero) | MAE (non-zero) |",
        "|----------------|--------|-------|--------|-------------|------|-----|-----|-----------|------------|-----------------|",
    ]
    for r in payload.get("rows", []):
        diag = "yes" if r.get("is_diagnostic_only") else "—"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r.get("benchmark_mode", ""),
                r.get("target_mode", ""),
                r.get("model_name", ""),
                r.get("model_family", ""),
                diag,
                r.get("rmse") if r.get("rmse") is not None else "—",
                r.get("mae") if r.get("mae") is not None else "—",
                r.get("r2") if r.get("r2") is not None else "—",
                r.get("test_zero_rate") if r.get("test_zero_rate") is not None else "—",
                r.get("mae_zero_target") if r.get("mae_zero_target") is not None else "—",
                r.get("mae_nonzero_target") if r.get("mae_nonzero_target") is not None else "—",
            )
        )
    lines.extend(["", "## Missing benchmark runs", ""])
    for m in payload.get("missing", []):
        lines.append(f"- **{m.get('benchmark_mode')}** / **{m.get('model_name')}**: {m.get('note', '')}")
    if payload.get("interpretation_notes"):
        lines.extend(["", "## Interpretation notes", ""])
        for k, v in payload["interpretation_notes"].items():
            lines.append(f"- **{k}**: {v}")
    lines.extend(["", "---", payload.get("notes", "")])
    return "\n".join(lines)


def ablation_to_md(payload: dict[str, Any]) -> str:
    """Render ablation_review payload as Markdown."""
    lines = [
        "# " + payload.get("report_name", "Ablation review"),
        "",
        f"**Generated:** {payload.get('generated_at', '')}",
        f"**Full model:** {payload.get('full_model_experiment', '')} (available: {payload.get('full_model_available', False)})",
        "",
        "## Ablations",
        "",
    ]
    if not payload.get("ablations"):
        hint = payload.get("empty_hint") or (
            "No ablation runs found. To populate: run train and evaluate for each config in "
            "configs/experiments/ablations/, or use: make run-temporal-h2-ablations (after make run-temporal-h2)."
        )
        lines.append(hint)
        lines.append("")
        return "\n".join(lines)
    lines.append("| Ablation | Type | Tag | Features removed | RMSE | MAE | R² | Δ RMSE | Δ MAE | Δ R² | Interpretation |")
    lines.append("|----------|------|-----|------------------|------|-----|-----|--------|-------|------|-----------------|")
    for a in payload.get("ablations", []):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                a.get("ablation_name", ""),
                a.get("ablation_type", ""),
                a.get("interpretation_tag", ""),
                ", ".join(a.get("features_removed", []))[:36],
                a.get("rmse") if a.get("rmse") is not None else "—",
                a.get("mae") if a.get("mae") is not None else "—",
                a.get("r2") if a.get("r2") is not None else "—",
                a.get("delta_rmse") if a.get("delta_rmse") is not None else "—",
                a.get("delta_mae") if a.get("delta_mae") is not None else "—",
                a.get("delta_r2") if a.get("delta_r2") is not None else "—",
                (a.get("interpretation") or "")[:40],
            )
        )
    return "\n".join(lines)


def run_benchmark_analysis(
    artifacts_root: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load all metrics, build comparison and ablation artifacts, write JSON + MD.
    Returns summary dict with paths written and counts.
    """
    if out_dir is None:
        out_dir = artifacts_root / "reports"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = load_all_metrics(artifacts_root)
    comparison = build_benchmark_comparison(metrics_list)
    ablation = build_ablation_review(metrics_list)

    comparison_json = out_dir / "benchmark_comparison.json"
    comparison_md = out_dir / "benchmark_comparison.md"
    ablation_json = out_dir / "ablation_review.json"
    ablation_md = out_dir / "ablation_review.md"

    save_json(comparison, comparison_json)
    comparison_md.write_text(comparison_to_md(comparison), encoding="utf-8")
    save_json(ablation, ablation_json)
    ablation_md.write_text(ablation_to_md(ablation), encoding="utf-8")

    return {
        "metrics_loaded": len(metrics_list),
        "comparison_rows": len(comparison.get("rows", [])),
        "comparison_missing": len(comparison.get("missing", [])),
        "ablation_count": len(ablation.get("ablations", [])),
        "written": {
            "benchmark_comparison.json": str(comparison_json),
            "benchmark_comparison.md": str(comparison_md),
            "ablation_review.json": str(ablation_json),
            "ablation_review.md": str(ablation_md),
        },
    }
