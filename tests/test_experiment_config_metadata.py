"""Tests that experiment configs support optional benchmark and ablation metadata."""

from pathlib import Path

import pytest

from scholarly_outcome_prediction.settings import load_experiment_config


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_experiment_config_with_benchmark_metadata(project_root: Path) -> None:
    """Config with benchmark block loads and exposes benchmark_mode, model_family, is_diagnostic_model."""
    path = project_root / "configs" / "experiments" / "baseline_representative.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    assert getattr(cfg, "benchmark", None) is not None
    assert cfg.benchmark.benchmark_mode == "representative_proxy"
    assert cfg.benchmark.model_family == "trivial_baseline"
    assert cfg.benchmark.is_diagnostic_model is False


def test_load_ablation_config_with_ablation_metadata(project_root: Path) -> None:
    """Ablation config with ablation block loads; features_removed is the authoritative source."""
    path = project_root / "configs" / "experiments" / "ablations" / "xgb_temporal_h2_no_publication_year.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    assert getattr(cfg, "ablation", None) is not None
    assert cfg.ablation.name == "no_publication_year"
    assert cfg.ablation.features_removed == ["publication_year"]
    assert cfg.ablation.ablation_type == "coarse"


def test_load_diagnostic_model_config(project_root: Path) -> None:
    """Year-conditioned config has is_diagnostic_model true."""
    path = project_root / "configs" / "experiments" / "year_conditioned_temporal_h2.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    assert cfg.benchmark.is_diagnostic_model is True
    assert cfg.benchmark.model_family == "diagnostic_baseline"
