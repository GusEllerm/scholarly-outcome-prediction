"""Row-level dataset overlap audit by work ID. Used to verify representative vs temporal distinctness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scholarly_outcome_prediction.utils.io import save_json

# Column used for row identity in processed parquets
DEFAULT_ID_COLUMN = "openalex_id"


def compute_overlap_report(
    path_left: Path,
    path_right: Path,
    id_column: str = DEFAULT_ID_COLUMN,
    label_left: str | None = None,
    label_right: str | None = None,
) -> dict[str, Any]:
    """
    Compare two processed datasets by work ID. Returns a report with sizes, overlap counts,
    overlap rates from both directions, whether datasets are identical, and sample IDs.
    """
    path_left = Path(path_left)
    path_right = Path(path_right)
    label_left = label_left or path_left.stem
    label_right = label_right or path_right.stem

    df_left = pd.read_parquet(path_left)
    df_right = pd.read_parquet(path_right)

    if id_column not in df_left.columns:
        return {
            "error": f"Left dataset missing id column '{id_column}'",
            "path_left": str(path_left),
            "path_right": str(path_right),
            "label_left": label_left,
            "label_right": label_right,
        }
    if id_column not in df_right.columns:
        return {
            "error": f"Right dataset missing id column '{id_column}'",
            "path_left": str(path_left),
            "path_right": str(path_right),
            "label_left": label_left,
            "label_right": label_right,
        }

    ids_left = set(df_left[id_column].dropna().astype(str))
    ids_right = set(df_right[id_column].dropna().astype(str))

    n_left = len(ids_left)
    n_right = len(ids_right)
    overlap = ids_left & ids_right
    n_overlap = len(overlap)
    only_left = ids_left - ids_right
    only_right = ids_right - ids_left

    overlap_rate_from_left = (n_overlap / n_left * 100.0) if n_left else 0.0
    overlap_rate_from_right = (n_overlap / n_right * 100.0) if n_right else 0.0
    identical = n_left == n_right == n_overlap and n_left > 0

    # Small samples for human inspection (sorted for determinism)
    sample_overlap = sorted(overlap)[:20] if overlap else []
    sample_only_left = sorted(only_left)[:20] if only_left else []
    sample_only_right = sorted(only_right)[:20] if only_right else []

    report: dict[str, Any] = {
        "path_left": str(path_left),
        "path_right": str(path_right),
        "label_left": label_left,
        "label_right": label_right,
        "id_column": id_column,
        "size_left": n_left,
        "size_right": n_right,
        "overlap_count": n_overlap,
        "overlap_rate_from_left_pct": round(overlap_rate_from_left, 2),
        "overlap_rate_from_right_pct": round(overlap_rate_from_right, 2),
        "identical": identical,
        "only_in_left_count": len(only_left),
        "only_in_right_count": len(only_right),
        "sample_overlap_ids": sample_overlap,
        "sample_only_in_left_ids": sample_only_left,
        "sample_only_in_right_ids": sample_only_right,
    }
    return report


def run_overlap_audit(
    path_left: Path,
    path_right: Path,
    out_dir: Path,
    id_column: str = DEFAULT_ID_COLUMN,
    label_left: str | None = None,
    label_right: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """
    Compute overlap report and write JSON + Markdown to out_dir.
    Returns (report, json_path, md_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = compute_overlap_report(
        path_left=path_left,
        path_right=path_right,
        id_column=id_column,
        label_left=label_left,
        label_right=label_right,
    )

    if "error" in report:
        json_path = out_dir / "dataset_overlap_audit.json"
        save_json(report, json_path)
        md_path = out_dir / "dataset_overlap_audit.md"
        md_path.write_text(
            f"# Dataset overlap audit\n\n**Error**: {report['error']}\n",
            encoding="utf-8",
        )
        return report, json_path, md_path

    slug = f"{report['label_left']}_vs_{report['label_right']}".replace(" ", "_")
    json_path = out_dir / f"{slug}_overlap_audit.json"
    md_path = out_dir / f"{slug}_overlap_audit.md"
    save_json(report, json_path)
    md_lines = _overlap_report_md(report)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report, json_path, md_path


def _overlap_report_md(report: dict[str, Any]) -> list[str]:
    """Human-readable Markdown for overlap report."""
    lines = [
        "# Dataset overlap audit",
        "",
        f"- **Left**: {report['label_left']} (`{report['path_left']}`)",
        f"- **Right**: {report['label_right']} (`{report['path_right']}`)",
        f"- **ID column**: {report['id_column']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Size (left) | {report['size_left']} |",
        f"| Size (right) | {report['size_right']} |",
        f"| Overlap count | {report['overlap_count']} |",
        f"| Overlap rate (from left) | {report['overlap_rate_from_left_pct']}% |",
        f"| Overlap rate (from right) | {report['overlap_rate_from_right_pct']}% |",
        f"| Identical | {report['identical']} |",
        f"| Only in left | {report['only_in_left_count']} |",
        f"| Only in right | {report['only_in_right_count']} |",
        "",
        "## Sample IDs",
        "",
        "Overlap (first 20):",
    ]
    for sid in report.get("sample_overlap_ids", [])[:20]:
        lines.append(f"- {sid}")
    lines.extend([
        "",
        "Only in left (first 20):",
    ])
    for sid in report.get("sample_only_in_left_ids", [])[:20]:
        lines.append(f"- {sid}")
    lines.extend([
        "",
        "Only in right (first 20):",
    ])
    for sid in report.get("sample_only_in_right_ids", [])[:20]:
        lines.append(f"- {sid}")
    return lines
