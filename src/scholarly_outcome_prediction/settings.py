"""Load and validate configs (data and experiment) using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from scholarly_outcome_prediction.utils.io import load_yaml


# --- Target semantics ---
# proxy: current bootstrap target (e.g. present-day cited_by_count)
# research: future fixed-horizon scholarly outcome (not fully implemented yet)
TARGET_MODE_PROXY = "proxy"
TARGET_MODE_RESEARCH = "research"


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


class TargetConfig(BaseModel):
    """Target variable, transform, and semantics (proxy vs research)."""

    name: str = "cited_by_count"
    transform: str | None = "log1p"  # "log1p" or None
    # proxy = bootstrap proxy (e.g. current cited_by_count); research = fixed-horizon outcome (future)
    target_mode: Literal["proxy", "research"] = "proxy"


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


class EvaluationConfig(BaseModel):
    """Evaluation metrics to compute."""

    metrics: list[str] = Field(default_factory=lambda: ["rmse", "mae", "r2"])


class ExperimentConfig(BaseModel):
    """Full experiment config: dataset, task, target, features, model, eval."""

    experiment_name: str = "experiment"
    task_type: str = "regression"  # regression | classification (classification not fully wired)
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    features: FeatureListsConfig = Field(default_factory=FeatureListsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_data_config(path: Path) -> DataConfig:
    """Load and validate data config from YAML."""
    raw = load_yaml(path)
    return DataConfig.model_validate(raw)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate experiment config from YAML."""
    raw = load_yaml(path)
    return ExperimentConfig.model_validate(raw)
