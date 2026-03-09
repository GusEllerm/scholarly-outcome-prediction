from __future__ import annotations

from pathlib import Path

from scholarly_outcome_prediction.cli import run_pipeline_from_configs


def main() -> None:
  """Convenience entrypoint to run the example pipeline."""
  project_root = Path(__file__).resolve().parents[1]
  data_config = project_root / "configs" / "data" / "openalex_sample_100.yaml"
  baseline_config = project_root / "configs" / "experiments" / "baseline_regression.yaml"
  xgb_config = project_root / "configs" / "experiments" / "xgb_regression.yaml"

  run_pipeline_from_configs(
    data_config_path=data_config,
    baseline_config_path=baseline_config,
    xgb_config_path=xgb_config,
  )


if __name__ == "__main__":
  main()

