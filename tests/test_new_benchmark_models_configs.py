"""Tests that new benchmark-expansion models have valid strict configs and explicit metadata."""

from pathlib import Path

import pytest

from scholarly_outcome_prediction.settings import load_current_experiment_config

# One representative config per active benchmark model (strict loader + benchmark metadata). Excludes experimental/non-benchmark models (e.g. tweedie).
NEW_MODEL_CONFIGS = [
    "elastic_net_representative",
    "extra_trees_representative",
    "hist_gradient_boosting_representative",
]


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("experiment_name", NEW_MODEL_CONFIGS)
def test_new_model_config_loads_strict_and_has_benchmark_metadata(
    project_root: Path, experiment_name: str
) -> None:
    """Each new model has at least one config that passes strict validation with explicit benchmark block."""
    config_path = project_root / "configs" / "experiments" / f"{experiment_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")
    cfg = load_current_experiment_config(config_path)
    assert cfg.benchmark is not None
    assert cfg.benchmark.benchmark_mode == "representative_proxy"
    assert cfg.benchmark.model_family in ("linear_baseline", "tree_model")
    assert cfg.benchmark.is_diagnostic_model is False
    assert cfg.model.name in ("elastic_net", "extra_trees", "hist_gradient_boosting")
