"""Unified benchmark comparison and ablation review from metrics artifacts.

Classification uses explicit metadata from run artifacts when present (benchmark_mode,
model_family, is_diagnostic_model, ablation_name, ablation_features_removed, ablation_type).
Legacy compatibility: for older metrics JSONs that lack these fields, inference from
experiment_name/dataset_id and fallback maps is used and labeled as legacy_inferred.
Current job configs must not rely on this; they must set benchmark (and ablation) in YAML.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scholarly_outcome_prediction.logging_utils import get_logger
from scholarly_outcome_prediction.utils.io import load_json, save_json

logger = get_logger(__name__)

# Canonical benchmark modes and target modes for comparison table
BENCHMARK_MODES = ["representative_proxy", "temporal_proxy", "representative_h2", "temporal_h2"]
ABLATION_FULL_EXPERIMENT = "xgb_temporal_h2"
# Legacy compatibility only: used when reading older metrics artifacts that lack
# explicit ablation_features_removed. Current runs must set ablation in experiment config.
ABLATION_FEATURES_REMOVED_FALLBACK: dict[str, list[str]] = {
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
# Legacy compatibility only: when reading older metrics that lack explicit model_family / is_diagnostic_model.
# Values are normalized to the same canonical snake_case vocabulary used by explicit metadata.
MODEL_FAMILY_FALLBACK: dict[str, str] = {
    "baseline": "trivial_baseline",
    "ridge": "linear_baseline",
    "year_conditioned": "diagnostic_baseline",
    "hurdle": "hurdle_baseline",
    "xgboost": "tree_model",
}
DIAGNOSTIC_ONLY_MODELS_FALLBACK = {"year_conditioned"}

# Backward compatibility for tests / external code that referenced these
ABLATION_FEATURES_REMOVED = ABLATION_FEATURES_REMOVED_FALLBACK


def _legacy_infer_benchmark_mode(experiment_name: str, data: dict[str, Any]) -> str | None:
    """Legacy compatibility: infer benchmark mode from experiment_name and dataset_id for older artifacts only."""
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


def _resolve_benchmark_mode(exp: str, data: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve benchmark mode: explicit if present, else legacy inference for older artifacts. Returns (mode, classification_source)."""
    explicit = data.get("benchmark_mode")
    if explicit and explicit.strip():
        return explicit.strip(), "explicit"
    inferred = _legacy_infer_benchmark_mode(exp, data)
    if inferred is not None:
        logger.debug("benchmark_mode legacy_inferred for %s (missing explicit benchmark_mode)", exp)
        return inferred, "legacy_inferred"
    return None, "legacy_inferred"


def _infer_target_mode(experiment_name: str, data: dict[str, Any]) -> str:
    """Infer target mode: proxy | h2 (calendar_horizon)."""
    if data.get("target_mode") == "calendar_horizon":
        return "h2"
    name = (experiment_name or "").lower()
    if "_h2" in name or "h2" in name:
        return "h2"
    return "proxy"


def _resolve_model_family(model_name: str, data: dict[str, Any]) -> tuple[str, str]:
    """Resolve model_family: explicit if present, else legacy fallback for older artifacts. Returns (family, classification_source)."""
    explicit = data.get("model_family")
    if explicit and explicit.strip():
        return explicit.strip(), "explicit"
    family = MODEL_FAMILY_FALLBACK.get(model_name, "other")
    logger.debug("model_family legacy_inferred for %s (missing explicit model_family)", model_name)
    return family, "legacy_inferred"


def _resolve_is_diagnostic(model_name: str, data: dict[str, Any]) -> tuple[bool, str]:
    """Resolve is_diagnostic: explicit if present, else legacy fallback for older artifacts. Returns (is_diagnostic, classification_source)."""
    explicit = data.get("is_diagnostic_model")
    if explicit is not None:
        return bool(explicit), "explicit"
    return model_name in DIAGNOSTIC_ONLY_MODELS_FALLBACK, "legacy_inferred"


def _is_ablation_from_data(data: dict[str, Any], experiment_name: str) -> bool:
    """True if this run is an ablation: has explicit ablation_name or matches legacy naming (older artifacts only)."""
    if data.get("ablation_name"):
        return True
    if not experiment_name or not experiment_name.startswith("xgb_temporal_h2_"):
        return False
    suffix = experiment_name.replace("xgb_temporal_h2_", "")
    return suffix in ABLATION_FEATURES_REMOVED_FALLBACK


def _ablation_name_from_data(data: dict[str, Any], experiment_name: str) -> str | None:
    """Return ablation name: explicit if present, else from experiment_name suffix (legacy compatibility)."""
    explicit = data.get("ablation_name")
    if explicit and explicit.strip():
        return explicit.strip()
    if not experiment_name or not experiment_name.startswith("xgb_temporal_h2_"):
        return None
    suffix = experiment_name.replace("xgb_temporal_h2_", "")
    return suffix if suffix in ABLATION_FEATURES_REMOVED_FALLBACK else None


def _ablation_features_removed_from_data(data: dict[str, Any], ablation_name: str | None) -> list[str]:
    """Explicit ablation_features_removed from run artifact is authoritative; else legacy fallback for older artifacts."""
    explicit = data.get("ablation_features_removed")
    if explicit is not None and isinstance(explicit, list):
        return list(explicit)
    if ablation_name:
        return ABLATION_FEATURES_REMOVED_FALLBACK.get(ablation_name, [])
    return []


def _ablation_type_from_data(data: dict[str, Any], ablation_name: str | None) -> str:
    """Resolve ablation_type: explicit if present, else coarse vs numeric_fine from name."""
    explicit = data.get("ablation_type")
    if explicit and explicit.strip():
        return explicit.strip()
    if ablation_name and ablation_name in ABLATION_TYPE_COARSE:
        return "coarse"
    return "numeric_fine"


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
    """Build unified comparison: rows per (benchmark_mode, model), plus missing list. Prefers explicit metadata."""
    generated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for data in metrics_list:
        exp = data.get("experiment_name") or data.get("_metrics_file", "").replace(".json", "")
        bm, bm_source = _resolve_benchmark_mode(exp, data)
        if bm is None:
            continue
        model = data.get("model_name", "unknown")
        key = (bm, model)
        if key in seen:
            continue
        seen.add(key)
        model_family, family_source = _resolve_model_family(model, data)
        is_diag, diag_source = _resolve_is_diagnostic(model, data)
        row: dict[str, Any] = {
            "benchmark_mode": bm,
            "benchmark_mode_source": bm_source,
            "target_mode": _infer_target_mode(exp, data),
            "split_kind": data.get("split_kind"),
            "model_name": model,
            "model_family": model_family,
            "model_family_source": family_source,
            "is_diagnostic_only": is_diag,
            "is_diagnostic_only_source": diag_source,
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
        "notes": "Eligibility filtering and target semantics (e.g. horizon) apply per run; see run metadata in each metrics JSON. Models with is_diagnostic_only=true are for interpretation only, not as primary baselines. Rows include benchmark_mode_source, model_family_source, is_diagnostic_only_source: 'explicit' from run metadata or 'legacy_inferred' for older artifacts only.",
    }


def build_ablation_review(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Build ablation comparison: each ablation vs full xgb_temporal_h2, with deltas. Uses explicit ablation metadata when present."""
    generated_at = datetime.now(timezone.utc).isoformat()
    full_metrics: dict[str, Any] | None = None
    for data in metrics_list:
        if (data.get("experiment_name") or "").strip() == ABLATION_FULL_EXPERIMENT:
            full_metrics = data
            break
    ablations: list[dict[str, Any]] = []
    for data in metrics_list:
        exp = data.get("experiment_name") or ""
        if not _is_ablation_from_data(data, exp):
            continue
        name = _ablation_name_from_data(data, exp)
        if name is None:
            continue
        features_removed = _ablation_features_removed_from_data(data, name)
        ablation_type = _ablation_type_from_data(data, name)
        row: dict[str, Any] = {
            "ablation_name": name,
            "experiment_name": exp,
            "features_removed": features_removed,
            "features_removed_source": "explicit" if data.get("ablation_features_removed") is not None else "legacy_inferred",
            "ablation_type": ablation_type,
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

    def _fmt(v: Any) -> Any:
        """Format numeric values to 4 decimal places for readability."""
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return v if v is not None else "—"

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
                ", ".join(a.get("features_removed", [])),
                _fmt(a.get("rmse")),
                _fmt(a.get("mae")),
                _fmt(a.get("r2")),
                _fmt(a.get("delta_rmse")),
                _fmt(a.get("delta_mae")),
                _fmt(a.get("delta_r2")),
                a.get("interpretation") or "",
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
