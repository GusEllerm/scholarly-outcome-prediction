#!/usr/bin/env python3
"""
Generate diagnostics artifacts for pipeline transparency.

Usage (from project root):
  uv run python scripts/generate_diagnostics.py
  uv run python scripts/generate_diagnostics.py --processed data/processed/openalex_representative_articles_1000.parquet
  uv run python scripts/generate_diagnostics.py --processed data/processed/openalex_temporal_articles_1000.parquet --dataset-id openalex_temporal_articles_1000

Each output JSON is stamped with dataset_id, run_id (from --run-id or processed path stem), and generated_at
so diagnostics reflect the current run rather than stale files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scholarly_outcome_prediction.diagnostics.generate_all import generate_all_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pipeline diagnostics")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Path to processed parquet (default: root/data/processed/openalex_pilot.parquet)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset ID for stamping (default: stem of --processed path)",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help="Path to configs dir (default: root/configs)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Path to artifacts dir (default: root/artifacts)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for diagnostics (default: root/artifacts/diagnostics)",
    )
    args = parser.parse_args()
    root = args.root
    processed_path = args.processed or root / "data" / "processed" / "openalex_pilot.parquet"
    processed_path = processed_path if processed_path.is_absolute() else root / processed_path
    dataset_id = args.dataset_id or (processed_path.stem if processed_path else None)

    out_dir = generate_all_diagnostics(
        root,
        processed_path,
        dataset_id=dataset_id,
        out_dir=args.out_dir,
        configs_dir=args.configs,
        artifacts_root=args.artifacts,
        include_design_trace=True,
    )
    print(f"Diagnostics written to {out_dir} (dataset_id={dataset_id})")
    print("  component_inventory.json, pipeline_trace_design.json, dataset_profile.json,")
    print("  missingness_summary.csv, feature_usage_report.json, run_artifact_audit.json,")
    print("  preprocessing_leakage_audit.json")


if __name__ == "__main__":
    main()
