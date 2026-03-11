"""Load and validate configs (data and experiment) using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scholarly_outcome_prediction.utils.io import load_yaml

# Error message used when current job config lacks required benchmark metadata
BENCHMARK_REQUIRED_HINT = (
    "Current train/evaluate jobs require a 'benchmark' block with benchmark_mode, model_family, and is_diagnostic_model. "
    "Add a benchmark block to your experiment YAML. See docs/job_yml_schema.md and docs/adding_a_model.md."
)
ABLATION_FEATURES_REQUIRED_HINT = (
    "When 'ablation' is present, 'name' and 'features_removed' (non-empty list) are required. "
    "See docs/job_yml_schema.md."
)


# --- Target semantics ---
# proxy: current bootstrap target (e.g. present-day cited_by_count)
# research: future fixed-horizon scholarly outcome (not fully implemented yet)
# calendar_horizon: citations summed over calendar years from counts_by_year
TARGET_MODE_PROXY = "proxy"
TARGET_MODE_RESEARCH = "research"
TARGET_MODE_CALENDAR_HORIZON = "calendar_horizon"


class DataConfig(BaseModel):
    """Data acquisition config (e.g. OpenAlex sample)."""

    dataset_name: str
    seed: int = 42
    sample_size: int = Field(..., gt=0, le=100000)
    from_publication_date: str = "2018-01-01"
    to_publication_date: str = "2018-12-31"
    fields: list[str] = Field(default_factory=list)
    output_path: str = "data/raw/openalex_sample.jsonl"
    # Optional: restrict work types (e.g. ["article"] for OpenAlex)
    work_types: list[str] | None = None
    # Optional: sort for API. Do NOT use for representative sampling (causes oldest-first slice).
    sort: str | None = None
    # When True, fetch per-year and combine so the sample spans the full date range.
    stratify_by_year: bool = False
    # When True with stratify_by_year, use OpenAlex sample+seed per year for within-year random sampling (representative).
    # When False with stratify_by_year, use cursor paging (temporal; order not randomized within year).
    use_random_sample: bool = False


class SplitConfig(BaseModel):
    """Train/test split settings. random = shuffle split; time = by year/date."""

    split_kind: str = "random"  # "random" | "time"
    test_size: float = Field(0.2, ge=0.0, lt=1.0)
    random_state: int = 42
    time_column: str | None = None  # e.g. "publication_year" for time-based split
    # For time split: explicit year boundaries (train = year <= train_year_end, test = year >= test_year_start).
    train_year_end: int | None = None
    test_year_start: int | None = None

    @model_validator(mode="after")
    def time_split_requires_time_column(self) -> "SplitConfig":
        if self.split_kind == "time":
            if not self.time_column or not str(self.time_column).strip():
                raise ValueError(
                    "split.split_kind is 'time' but split.time_column is missing or empty. "
                    "Set split.time_column (e.g. 'publication_year'). See docs/job_yml_schema.md."
                )
        return self


class TargetConfig(BaseModel):
    """Target variable, transform, and semantics (proxy vs calendar_horizon vs research)."""

    name: str = "cited_by_count"
    transform: str | None = "log1p"  # "log1p" or None
    # proxy = bootstrap proxy (e.g. current cited_by_count); calendar_horizon = from counts_by_year
    target_mode: Literal["proxy", "research", "calendar_horizon"] = "proxy"
    # For calendar_horizon: source field (counts_by_year), horizon length, include publication year
    source: str | None = None  # e.g. "counts_by_year"
    horizon_years: int | None = None  # e.g. 2
    include_publication_year: bool = True  # True = citations_within_H; False = citations_in_next_H

    @model_validator(mode="after")
    def calendar_horizon_requires_source_and_horizon(self) -> "TargetConfig":
        if self.target_mode == "calendar_horizon":
            if not self.source or not str(self.source).strip():
                raise ValueError(
                    "target_mode is 'calendar_horizon' but target.source is missing or empty. "
                    "Set target.source (e.g. 'counts_by_year'). See docs/job_yml_schema.md."
                )
            if self.horizon_years is None:
                raise ValueError(
                    "target_mode is 'calendar_horizon' but target.horizon_years is missing. "
                    "Set target.horizon_years (e.g. 2). See docs/job_yml_schema.md."
                )
        return self


class DataPathsConfig(BaseModel):
    """Paths to processed dataset and optional dataset identifier for run metadata."""

    processed_path: str = "data/processed/openalex_sample_100.parquet"
    dataset_id: str | None = None  # e.g. dataset_name from data config, for artifact metadata


class FeatureListsConfig(BaseModel):
    """Numeric and categorical feature names (metadata features only for now)."""

    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model name and hyperparameters."""

    name: str = "baseline"
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetadataConfig(BaseModel):
    """Explicit benchmark metadata for run artifacts and comparison. Required for current train/evaluate jobs."""

    benchmark_mode: str | None = None  # e.g. representative_proxy, temporal_proxy, representative_h2, temporal_h2
    model_family: str | None = None  # e.g. trivial_baseline, linear_baseline, tree_model, hurdle_baseline, diagnostic_baseline
    # Note: is_diagnostic_model remains defaulted here for backward compatibility when using the non-strict loader.
    # Strict current-job validation additionally checks that the field was explicitly present in the YAML.
    is_diagnostic_model: bool = False  # True = diagnostic-only (e.g. year_conditioned), not a primary comparator

    @model_validator(mode="after")
    def reject_empty_benchmark_identifiers(self) -> "BenchmarkMetadataConfig":
        for field in ("benchmark_mode", "model_family"):
            v = getattr(self, field)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(
                    f"benchmark.{field} must be a non-empty string when set. "
                    f"Got: {v!r}. {BENCHMARK_REQUIRED_HINT}"
                )
        return self


class AblationConfig(BaseModel):
    """Ablation experiment metadata. Single source of truth for features_removed in reports."""

    name: str  # e.g. no_publication_year
    features_removed: list[str] = Field(default_factory=list)
    ablation_type: str | None = None  # coarse | numeric_fine; optional, can be inferred from name if needed

    @model_validator(mode="after")
    def ablation_requires_name_and_features(self) -> "AblationConfig":
        if not self.name or not str(self.name).strip():
            raise ValueError(
                "ablation.name must be non-empty. " + ABLATION_FEATURES_REQUIRED_HINT
            )
        if not self.features_removed:
            raise ValueError(
                "ablation.features_removed must be a non-empty list when ablation block is present. "
                + ABLATION_FEATURES_REQUIRED_HINT
            )
        return self


class EvaluationConfig(BaseModel):
    """Evaluation metrics to compute."""

    metrics: list[str] = Field(default_factory=lambda: ["rmse", "mae", "r2"])


class ExperimentConfig(BaseModel):
    """Full experiment config: dataset, task, target, features, model, eval. Benchmark required for current train/evaluate jobs."""

    experiment_name: str = "experiment"
    task_type: str = "regression"  # regression | classification (classification not fully wired)
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    features: FeatureListsConfig = Field(default_factory=FeatureListsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmark: BenchmarkMetadataConfig | None = None  # required for current jobs; when set, carried into metrics
    ablation: AblationConfig | None = None  # when set, this run is an ablation; name and features_removed required


def validate_current_job_experiment_config(cfg: ExperimentConfig, raw: dict[str, Any] | None = None) -> None:
    """
    Validate that config has required benchmark (and ablation) metadata for current train/evaluate jobs.
    Raises ValueError with migration hint if not. Use after load_experiment_config for train/evaluate paths.
    """
    if cfg.benchmark is None:
        raise ValueError(
            f"Experiment config '{cfg.experiment_name}' is missing required 'benchmark' block. "
            + BENCHMARK_REQUIRED_HINT
        )
    if not (cfg.benchmark.benchmark_mode and str(cfg.benchmark.benchmark_mode).strip()):
        raise ValueError(
            f"Experiment config '{cfg.experiment_name}': benchmark.benchmark_mode must be non-empty. "
            + BENCHMARK_REQUIRED_HINT
        )
    if not (cfg.benchmark.model_family and str(cfg.benchmark.model_family).strip()):
        raise ValueError(
            f"Experiment config '{cfg.experiment_name}': benchmark.model_family must be non-empty. "
            + BENCHMARK_REQUIRED_HINT
        )
    # For current jobs, benchmark.is_diagnostic_model must be explicitly provided in YAML (True or False),
    # rather than silently defaulting to False.
    if raw is not None and isinstance(raw, dict):
        raw_benchmark = raw.get("benchmark")
        if isinstance(raw_benchmark, dict) and "is_diagnostic_model" not in raw_benchmark:
            raise ValueError(
                f"Experiment config '{cfg.experiment_name}' is missing required field "
                "'benchmark.is_diagnostic_model'. Current benchmarked jobs must specify this "
                "explicitly as true or false. " + BENCHMARK_REQUIRED_HINT
            )
    if cfg.ablation is not None:
        if not (cfg.ablation.name and str(cfg.ablation.name).strip()):
            raise ValueError(
                f"Experiment config '{cfg.experiment_name}' has ablation block but ablation.name is empty. "
                + ABLATION_FEATURES_REQUIRED_HINT
            )
        if not cfg.ablation.features_removed:
            raise ValueError(
                f"Experiment config '{cfg.experiment_name}' has ablation block but ablation.features_removed is empty. "
                + ABLATION_FEATURES_REQUIRED_HINT
            )


def load_data_config(path: Path) -> DataConfig:
    """Load and validate data config from YAML."""
    raw = load_yaml(path)
    return DataConfig.model_validate(raw)


def load_experiment_config(path: Path, *, strict_current_job: bool = False) -> ExperimentConfig:
    """
    Load and validate experiment config from YAML.
    When strict_current_job=True, requires benchmark block (and valid ablation if present) for train/evaluate.
    Use strict_current_job=True for train, evaluate, and run; use False for report-only or legacy artifact reading.
    """
    raw = load_yaml(path)
    cfg = ExperimentConfig.model_validate(raw)
    if strict_current_job:
        validate_current_job_experiment_config(cfg, raw)
    return cfg


def load_current_experiment_config(path: Path) -> ExperimentConfig:
    """Load experiment config for current train/evaluate. Fails if required benchmark or ablation metadata is missing."""
    return load_experiment_config(path, strict_current_job=True)
