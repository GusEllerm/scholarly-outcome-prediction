"""Dataset validation: year spread, type distribution, missingness, venue rate, citation realism. Fails loudly when invalid."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from scholarly_outcome_prediction.diagnostics.dataset_stats import compute_canonical_dataset_stats
from scholarly_outcome_prediction.utils.io import save_json

# Column used for work identity in processed data
WORK_ID_COLUMN = "openalex_id"


def _work_id_fingerprint(df: pd.DataFrame, id_column: str = WORK_ID_COLUMN) -> str | None:
    """Return a compact hash of sorted work IDs for provenance, or None if column missing/empty."""
    if id_column not in df.columns:
        return None
    ids = df[id_column].dropna().astype(str).sort_values().unique()
    if len(ids) == 0:
        return None
    blob = "|".join(ids)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _selection_strategy_summary(
    dataset_mode: str | None,
    generation_params: dict[str, Any] | None,
) -> str:
    """Human-readable summary of dataset semantics and acquisition (for provenance).
    dataset_mode is authoritative: representative = broad-sample / representative-oriented evaluation;
    temporal = forward-time generalization with time-ordered train/test split.
    """
    if dataset_mode == "representative":
        acq = "n/a"
        if generation_params:
            strat = generation_params.get("stratify_by_year")
            use_random = generation_params.get("use_random_sample")
            if strat and use_random:
                acq = "stratify_by_year=true, use_random_sample=true (within-year random sampling)"
            else:
                acq = f"stratify_by_year={strat}, use_random_sample={use_random}"
        return (
            "Representative dataset: intended to approximate a broad article sample. "
            "Random within-year sampling is acceptable; split strategy may be random or otherwise representative-oriented. "
            f"Acquisition: {acq}."
        )
    if dataset_mode == "temporal":
        acq = "n/a"
        if generation_params:
            strat = generation_params.get("stratify_by_year")
            use_random = generation_params.get("use_random_sample")
            if strat and use_random is False:
                acq = "stratify_by_year=true, use_random_sample=false (cursor per year, API order)"
            elif strat and use_random:
                acq = (
                    "stratify_by_year=true, use_random_sample=true (within-year random sample per year); "
                    "evaluation uses time-ordered train/test split."
                )
            else:
                acq = f"stratify_by_year={strat}, use_random_sample={use_random}"
        return (
            "Temporal dataset: intended for forward-time generalization. "
            "The key benchmark distinction is time-ordered evaluation (train on past, test on future). "
            "Do not interpret this dataset as a representative sample; split semantics define the benchmark. "
            f"Acquisition: {acq}."
        )
    if dataset_mode:
        acq = "n/a"
        if generation_params:
            acq = f"stratify_by_year={generation_params.get('stratify_by_year')}, use_random_sample={generation_params.get('use_random_sample')}"
        return f"{dataset_mode}: acquisition {acq}."
    if generation_params:
        return f"unspecified mode; stratify_by_year={generation_params.get('stratify_by_year')}, use_random_sample={generation_params.get('use_random_sample')}"
    return "unspecified"

# Severity for validation messages: error (fails), warning, informational, expected (e.g. article-only by config)
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFORMATIONAL = "informational"
SEVERITY_EXPECTED = "expected"


# Default thresholds (can be overridden by validation config)
DEFAULT_MIN_ROW_COUNT = 100
DEFAULT_MIN_YEARS_WITH_DATA = 2
DEFAULT_MAX_VENUE_MISSINGNESS_PCT = 95.0
# Representative mode: flag elite-only corpora (median citations above threshold suggests biased sample)
DEFAULT_MAX_MEDIAN_CITATIONS_REPRESENTATIVE = 500.0
DEFAULT_MIN_DISTINCT_VENUES_REPRESENTATIVE = 10


def _raw_venue_like_count(records: list[dict[str, Any]]) -> int:
    """Count raw records that have any venue-like field (source.display_name, raw_source_name, etc.)."""
    n = 0
    for r in records:
        if not isinstance(r, dict):
            continue
        pl = r.get("primary_location")
        if isinstance(pl, dict):
            if pl.get("raw_source_name"):
                n += 1
                continue
            src = pl.get("source")
            if isinstance(src, dict) and src.get("display_name"):
                n += 1
                continue
            if pl.get("display_name"):
                n += 1
                continue
        hv = r.get("host_venue")
        if isinstance(hv, dict) and (hv.get("display_name") or (hv.get("source") or {}).get("display_name")):
            n += 1
            continue
        locs = r.get("locations")
        if isinstance(locs, list) and locs and isinstance(locs[0], dict):
            src = (locs[0].get("source") or {})
            if isinstance(src, dict) and src.get("display_name"):
                n += 1
    return n


def validate_raw_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate raw OpenAlex records: row count, type distribution, year distribution, venue-like availability."""
    if not records:
        return {
            "row_count": 0,
            "type_counts": {},
            "publication_year_counts": {},
            "venue_like_available_count": 0,
            "venue_like_available_pct": 0.0,
            "errors": ["No records"],
        }
    type_counts: dict[str, int] = {}
    year_counts: dict[int, int] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        t = r.get("type")
        if t is not None:
            type_counts[str(t)] = type_counts.get(str(t), 0) + 1
        y = r.get("publication_year")
        if y is not None:
            try:
                yi = int(y)
                year_counts[yi] = year_counts.get(yi, 0) + 1
            except (TypeError, ValueError):
                pass
    venue_like = _raw_venue_like_count(records)
    n = len(records)
    return {
        "row_count": n,
        "type_counts": type_counts,
        "publication_year_counts": dict(sorted(year_counts.items())),
        "venue_like_available_count": venue_like,
        "venue_like_available_pct": round(100.0 * venue_like / n, 2) if n else 0.0,
        "publication_year_min": min(year_counts.keys()) if year_counts else None,
        "publication_year_max": max(year_counts.keys()) if year_counts else None,
    }


def _add_message(
    messages: list[dict[str, str]],
    severity: str,
    text: str,
) -> None:
    messages.append({"severity": severity, "text": text})


def validate_processed_dataset(
    df: pd.DataFrame,
    min_row_count: int = DEFAULT_MIN_ROW_COUNT,
    min_years_with_data: int = DEFAULT_MIN_YEARS_WITH_DATA,
    max_venue_missingness_pct: float = DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
    expected_year_min: int | None = None,
    expected_year_max: int | None = None,
    dataset_mode: str | None = None,
    expected_work_types: list[str] | None = None,
    max_median_citations_representative: float | None = DEFAULT_MAX_MEDIAN_CITATIONS_REPRESENTATIVE,
    min_distinct_venues_representative: int | None = DEFAULT_MIN_DISTINCT_VENUES_REPRESENTATIVE,
) -> dict[str, Any]:
    """
    Validate processed DataFrame using canonical stats. When expected_work_types is set (e.g. ["article"]),
    a single type matching that is treated as expected, not a warning.
    Messages include severity: error | warning | informational | expected.
    """
    stats = compute_canonical_dataset_stats(df)
    result: dict[str, Any] = {
        **stats,
        "passed": True,
        "errors": [],
        "warnings": [],
        "messages": [],
    }
    n = stats["row_count"]
    messages: list[dict[str, str]] = []

    if n < min_row_count:
        result["passed"] = False
        result["errors"].append(f"Row count {n} below minimum {min_row_count}")
        _add_message(messages, SEVERITY_ERROR, result["errors"][-1])

    year_info = stats.get("publication_year") or {}
    year_n_unique = year_info.get("n_unique", 0)
    if not year_info and "publication_year" in df.columns:
        result["errors"].append("publication_year is all null")
        _add_message(messages, SEVERITY_ERROR, result["errors"][-1])
    elif "publication_year" not in df.columns:
        result["errors"].append("publication_year column missing")
        _add_message(messages, SEVERITY_ERROR, result["errors"][-1])
    else:
        if year_n_unique < min_years_with_data:
            result["passed"] = False
            msg = f"Publication year has only {year_n_unique} distinct value(s); need at least {min_years_with_data}"
            result["errors"].append(msg)
            _add_message(messages, SEVERITY_ERROR, msg)
        if year_n_unique == 1:
            _add_message(messages, SEVERITY_WARNING, "Single year only; corpus may not be representative across time")
            result["warnings"].append(messages[-1]["text"])
        if expected_year_min is not None and (year_info.get("min") or 0) > expected_year_min:
            msg = f"Earliest year {year_info.get('min')} is after expected min {expected_year_min}"
            _add_message(messages, SEVERITY_WARNING, msg)
            result["warnings"].append(msg)
        if expected_year_max is not None and (year_info.get("max") or 0) < expected_year_max:
            msg = f"Latest year {year_info.get('max')} is before expected max {expected_year_max}"
            _add_message(messages, SEVERITY_WARNING, msg)
            result["warnings"].append(msg)

    type_counts = stats.get("type_counts") or {}
    if type_counts and len(type_counts) == 1:
        single_type = list(type_counts.keys())[0]
        if expected_work_types and set(type_counts.keys()) <= set(expected_work_types):
            _add_message(messages, SEVERITY_EXPECTED, f"Single type '{single_type}' matches config work_types {expected_work_types}")
        else:
            _add_message(messages, SEVERITY_WARNING, f"Single type only: {[single_type]}")
            result["warnings"].append(messages[-1]["text"])

    citation_dist = stats.get("citation_distribution") or {}
    if citation_dist and dataset_mode == "representative" and max_median_citations_representative is not None:
        med = citation_dist.get("median")
        if med is not None:
            if med > max_median_citations_representative:
                result["passed"] = False
                msg = (
                    f"Representative corpus median citations {med:.0f} exceeds threshold {max_median_citations_representative}; "
                    "corpus may be elite-only (check sampling)"
                )
                result["errors"].append(msg)
                _add_message(messages, SEVERITY_ERROR, msg)
            elif med > max_median_citations_representative * 0.5:
                msg = f"Median citations {med:.0f} is high for a representative corpus (threshold {max_median_citations_representative})"
                result["warnings"].append(msg)
                _add_message(messages, SEVERITY_WARNING, msg)

    venue_pct = stats.get("venue_name_non_null_pct", 0)
    venue_missing_pct = 100.0 - venue_pct
    if "venue_name" not in df.columns:
        result["passed"] = False
        result["errors"].append("venue_name column missing")
        _add_message(messages, SEVERITY_ERROR, result["errors"][-1])
    else:
        if venue_missing_pct > max_venue_missingness_pct:
            result["passed"] = False
            msg = f"venue_name missingness {venue_missing_pct:.1f}% exceeds threshold {max_venue_missingness_pct}%"
            result["errors"].append(msg)
            _add_message(messages, SEVERITY_ERROR, msg)
        elif venue_missing_pct > 80:
            result["warnings"].append(f"venue_name missingness is high: {venue_missing_pct:.1f}%")
            _add_message(messages, SEVERITY_WARNING, result["warnings"][-1])
        if dataset_mode == "representative" and min_distinct_venues_representative is not None:
            distinct_venues = stats.get("distinct_venue_count", 0)
            if distinct_venues < min_distinct_venues_representative:
                result["passed"] = False
                msg = (
                    f"Representative corpus has only {distinct_venues} distinct venues "
                    f"(minimum {min_distinct_venues_representative})"
                )
                result["errors"].append(msg)
                _add_message(messages, SEVERITY_ERROR, msg)

    result["citation_quantiles"] = citation_dist  # backward compat
    result["messages"] = messages
    return result


def run_validation_and_save(
    raw_records: list[dict[str, Any]] | None,
    df: pd.DataFrame,
    processed_path: Path,
    out_dir: Path,
    run_id: str | None = None,
    min_row_count: int = DEFAULT_MIN_ROW_COUNT,
    min_years_with_data: int = DEFAULT_MIN_YEARS_WITH_DATA,
    max_venue_missingness_pct: float = DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
    expected_year_min: int | None = None,
    expected_year_max: int | None = None,
    dataset_mode: str | None = None,
    expected_work_types: list[str] | None = None,
    max_median_citations_representative: float | None = DEFAULT_MAX_MEDIAN_CITATIONS_REPRESENTATIVE,
    min_distinct_venues_representative: int | None = DEFAULT_MIN_DISTINCT_VENUES_REPRESENTATIVE,
    source_config_path: str | Path | None = None,
    generation_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """
    Run raw + processed validation, merge results, save JSON and MD. Returns (merged_result, json_path, md_path).
    When run_id is provided, report is run-scoped (specific execution). When run_id is None, report is dataset-scoped
    and uses report_id and source_dataset_id only. Filename always uses source_dataset_id (processed_path.stem).
    Optional source_config_path and generation_params are recorded in provenance for dataset identity.
    """
    from scholarly_outcome_prediction.diagnostics.report_metadata import report_metadata

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dataset_id = processed_path.stem

    raw_result: dict[str, Any] = {}
    if raw_records is not None:
        raw_result = validate_raw_records(raw_records)
        raw_result["_source"] = "raw_records"

    proc_result = validate_processed_dataset(
        df,
        min_row_count=min_row_count,
        min_years_with_data=min_years_with_data,
        max_venue_missingness_pct=max_venue_missingness_pct,
        expected_year_min=expected_year_min,
        expected_year_max=expected_year_max,
        dataset_mode=dataset_mode,
        expected_work_types=expected_work_types,
        max_median_citations_representative=max_median_citations_representative,
        min_distinct_venues_representative=min_distinct_venues_representative,
    )
    proc_result["_source"] = "processed_dataframe"
    proc_result["processed_path"] = str(processed_path)

    merged_errors = list(proc_result.get("errors", []))
    if raw_result.get("errors"):
        merged_errors = merged_errors + list(raw_result["errors"])

    if run_id is not None:
        meta = report_metadata(
            report_scope="run",
            report_name="dataset_validation",
            run_id=run_id,
            dataset_id=source_dataset_id,
            source_dataset_path=str(processed_path),
            source_dataset_id=source_dataset_id,
            config_paths=None,
        )
    else:
        meta = report_metadata(
            report_scope="dataset",
            report_name="dataset_validation",
            report_id=f"{source_dataset_id}_dataset_validation",
            dataset_id=source_dataset_id,
            source_dataset_path=str(processed_path),
            source_dataset_id=source_dataset_id,
            config_paths=None,
        )

    # Provenance: identity and generation so two reports are distinguishable even when stats look similar
    work_id_fingerprint = _work_id_fingerprint(df)
    selection_strategy_summary = _selection_strategy_summary(dataset_mode, generation_params)
    provenance: dict[str, Any] = {
        "dataset_id": source_dataset_id,
        "processed_path": str(processed_path),
        "dataset_mode": dataset_mode,
        "work_id_fingerprint": work_id_fingerprint,
        "publication_year_coverage": (proc_result.get("publication_year") or {}).get("counts"),
    }
    if source_config_path is not None:
        provenance["source_config_path"] = str(source_config_path)
    if generation_params:
        provenance["generation_params"] = generation_params
    if selection_strategy_summary:
        provenance["selection_strategy_summary"] = selection_strategy_summary

    merged: dict[str, Any] = {
        **meta,
        "dataset_mode": dataset_mode,
        "expected_work_types": expected_work_types,
        "processed_path": str(processed_path),
        "provenance": provenance,
        "validated_at": meta["generated_at"],
        "raw": raw_result,
        "processed": proc_result,
        "passed": proc_result.get("passed", False) and len(merged_errors) == 0,
        "errors": merged_errors,
        "warnings": list(proc_result.get("warnings", [])),
        "messages": list(proc_result.get("messages", [])),
    }

    json_path = out_dir / f"{source_dataset_id}_dataset_validation.json"
    save_json(merged, json_path)

    # Markdown report with severity and provenance
    prov = merged.get("provenance", {})
    lines = [
        f"# Dataset validation: {source_dataset_id}",
        "",
        f"- **Report scope**: {meta.get('report_scope')}",
        f"- **Processed path**: {processed_path}",
        f"- **Dataset mode**: {dataset_mode or 'unspecified'}",
        f"- **Dataset ID**: {prov.get('dataset_id', source_dataset_id)}",
        f"- **Work ID fingerprint**: {prov.get('work_id_fingerprint') or 'n/a'}",
        f"- **Source config**: {prov.get('source_config_path') or 'n/a'}",
        f"- **Selection strategy**: {prov.get('selection_strategy_summary') or 'n/a'}",
        f"- **Passed**: {merged['passed']}",
        f"- **Validated at**: {merged['validated_at']}",
        "",
        "## Provenance",
        "",
        f"- generation_params: {prov.get('generation_params') or 'n/a'}",
        f"- publication_year coverage: {list((prov.get('publication_year_coverage') or {}).keys())[:10]}...",
        "",
        "## Messages (by severity)",
    ]
    for m in merged.get("messages", []):
        lines.append(f"- **{m.get('severity', 'info')}**: {m.get('text', '')}")
    lines.extend(["", "## Errors"])
    for e in merged["errors"]:
        lines.append(f"- {e}")
    lines.extend(["", "## Warnings"])
    for w in merged["warnings"]:
        lines.append(f"- {w}")
    lines.extend([
        "",
        "## Processed stats (canonical)",
        f"- Row count: {proc_result.get('row_count')}",
        f"- Publication year: {proc_result.get('publication_year')}",
        f"- Citation distribution: {proc_result.get('citation_distribution')}",
        f"- venue_name non-null %: {proc_result.get('venue_name_non_null_pct')}",
        f"- Distinct venues: {proc_result.get('distinct_venue_count')}",
        f"- Distinct primary_topic: {proc_result.get('distinct_primary_topic_count')}",
        f"- Distinct language: {proc_result.get('distinct_language_count')}",
        "",
        "## Raw stats (if available)",
    ])
    if raw_result:
        lines.append(f"- Row count: {raw_result.get('row_count')}")
        lines.append(f"- Venue-like available %: {raw_result.get('venue_like_available_pct')}")
    md_path = out_dir / f"{source_dataset_id}_dataset_validation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return merged, json_path, md_path
