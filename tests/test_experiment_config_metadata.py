"""Tests for experiment config validation: strict current-job path and semantic rules."""

from pathlib import Path

import pytest

from scholarly_outcome_prediction.settings import (
    load_experiment_config,
    load_current_experiment_config,
    validate_current_job_experiment_config,
    ExperimentConfig,
    BenchmarkMetadataConfig,
    AblationConfig,
)
from scholarly_outcome_prediction.utils.io import load_yaml


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_experiment_config_with_benchmark_metadata(project_root: Path) -> None:
    """Config with valid benchmark block loads; benchmark_mode, model_family, is_diagnostic_model are set."""
    path = project_root / "configs" / "experiments" / "baseline_representative.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    assert getattr(cfg, "benchmark", None) is not None
    assert cfg.benchmark.benchmark_mode == "representative_proxy"
    assert cfg.benchmark.model_family == "trivial_baseline"
    assert cfg.benchmark.is_diagnostic_model is False


def test_load_current_experiment_config_succeeds_with_benchmark(project_root: Path) -> None:
    """Strict loader accepts config that has required benchmark block."""
    path = project_root / "configs" / "experiments" / "baseline_representative.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_current_experiment_config(path)
    assert cfg.benchmark is not None
    assert cfg.benchmark.benchmark_mode == "representative_proxy"


def test_load_current_experiment_config_fails_without_benchmark(project_root: Path) -> None:
    """Strict loader fails when benchmark block is missing (current jobs must have benchmark)."""
    path = project_root / "configs" / "experiments" / "baseline_representative.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    # Build a config without benchmark by replacing with None
    cfg_no_benchmark = cfg.model_copy(update={"benchmark": None})
    with pytest.raises(ValueError, match="missing required 'benchmark' block"):
        validate_current_job_experiment_config(cfg_no_benchmark)


def test_strict_loader_requires_is_diagnostic_model_in_yaml(project_root: Path) -> None:
    """Strict validation fails when benchmark.is_diagnostic_model is missing from raw YAML."""
    path = project_root / "configs" / "experiments" / "baseline_representative.yaml"
    if not path.exists():
        pytest.skip("config not found")
    raw = load_yaml(path)
    # Simulate a config where is_diagnostic_model was omitted in YAML
    raw_benchmark = dict(raw.get("benchmark", {}))
    raw_benchmark.pop("is_diagnostic_model", None)
    raw_without_flag = {**raw, "benchmark": raw_benchmark}
    cfg = ExperimentConfig.model_validate(raw_without_flag)
    with pytest.raises(ValueError, match="benchmark.is_diagnostic_model"):
        validate_current_job_experiment_config(cfg, raw_without_flag)


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


def test_load_current_experiment_config_ablation_succeeds_with_benchmark(project_root: Path) -> None:
    """Strict loader accepts ablation config that has both benchmark and valid ablation block."""
    path = project_root / "configs" / "experiments" / "ablations" / "xgb_temporal_h2_no_publication_year.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_current_experiment_config(path)
    assert cfg.benchmark is not None
    assert cfg.ablation is not None
    assert cfg.ablation.features_removed == ["publication_year"]


def test_load_diagnostic_model_config(project_root: Path) -> None:
    """Year-conditioned config has is_diagnostic_model true."""
    path = project_root / "configs" / "experiments" / "year_conditioned_temporal_h2.yaml"
    if not path.exists():
        pytest.skip("config not found")
    cfg = load_experiment_config(path)
    assert cfg.benchmark.is_diagnostic_model is True
    assert cfg.benchmark.model_family == "diagnostic_baseline"


def test_benchmark_config_rejects_empty_benchmark_mode() -> None:
    """BenchmarkMetadataConfig rejects empty benchmark_mode when set."""
    with pytest.raises(ValueError, match="benchmark_mode must be a non-empty string"):
        BenchmarkMetadataConfig(
            benchmark_mode="",
            model_family="trivial_baseline",
            is_diagnostic_model=False,
        )


def test_benchmark_config_rejects_empty_model_family() -> None:
    """BenchmarkMetadataConfig rejects empty model_family when set."""
    with pytest.raises(ValueError, match="model_family must be a non-empty string"):
        BenchmarkMetadataConfig(
            benchmark_mode="representative_proxy",
            model_family="  ",
            is_diagnostic_model=False,
        )


def test_ablation_config_rejects_empty_name() -> None:
    """AblationConfig rejects empty name."""
    with pytest.raises(ValueError, match="ablation.name must be non-empty"):
        AblationConfig(name="", features_removed=["x"], ablation_type="coarse")


def test_ablation_config_rejects_empty_features_removed() -> None:
    """AblationConfig rejects empty features_removed list when ablation block is present."""
    with pytest.raises(ValueError, match="ablation.features_removed must be a non-empty list"):
        AblationConfig(name="no_x", features_removed=[], ablation_type="coarse")
