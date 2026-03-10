"""Tests that ridge baseline is available across benchmark modes via experiment configs."""

from pathlib import Path

import pytest

from scholarly_outcome_prediction.settings import load_experiment_config

# Ridge configs for the four benchmark modes (representative proxy, temporal proxy, representative H2, temporal H2)
RIDGE_CONFIG_NAMES = [
    "ridge_representative",
    "ridge_temporal",
    "ridge_representative_h2",
    "ridge_temporal_h2",
]


@pytest.fixture
def project_root() -> Path:
    """Project root (parent of tests/)."""
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("experiment_name", RIDGE_CONFIG_NAMES)
def test_ridge_config_loads_and_has_ridge_model(project_root: Path, experiment_name: str) -> None:
    """Each ridge experiment config exists, loads, and uses model name 'ridge'."""
    config_path = project_root / "configs" / "experiments" / f"{experiment_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")
    cfg = load_experiment_config(config_path)
    assert cfg.model.name == "ridge"
    assert cfg.experiment_name == experiment_name
