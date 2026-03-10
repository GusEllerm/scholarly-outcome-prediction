"""CLI: fetch, prepare, train, evaluate, run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
import typer

from scholarly_outcome_prediction.data import normalize_works_to_dataframe, train_test_split_df
from scholarly_outcome_prediction.evaluation import (
    build_run_metadata,
    compute_metrics,
    load_model_pipeline,
    save_metrics,
    save_model_pipeline,
)
from scholarly_outcome_prediction.features import build_feature_matrix, build_preprocessor
from scholarly_outcome_prediction.logging_utils import setup_logging, get_logger
from scholarly_outcome_prediction.models import get_model_builder
from scholarly_outcome_prediction.settings import load_data_config, load_experiment_config
from scholarly_outcome_prediction.utils.io import load_jsonl, read_parquet, write_parquet
from scholarly_outcome_prediction.utils.seeds import set_global_seed

from scholarly_outcome_prediction.acquisition import fetch_and_save
from scholarly_outcome_prediction.validation import run_validation_and_save

app = typer.Typer(help="Scholarly outcome prediction pipeline")
logger = get_logger(__name__)


def _ensure_project_cwd() -> Path:
    """Use current dir as project root (config paths are relative)."""
    return Path.cwd()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    setup_logging(level="DEBUG" if verbose else "INFO")
    if ctx.invoked_subcommand is None:
        typer.echo("Use --help to see commands: fetch, prepare, train, evaluate, run.")


@app.command()
def fetch(
    config: Path = typer.Option(..., "--config", "-c", path_type=Path, help="Data config YAML"),
) -> None:
    """Fetch OpenAlex sample and save raw JSONL."""
    root = _ensure_project_cwd()
    cfg = load_data_config(root / config)
    out_path = root / cfg.output_path
    fetch_and_save(
        output_path=out_path,
        sample_size=cfg.sample_size,
        from_publication_date=cfg.from_publication_date,
        to_publication_date=cfg.to_publication_date,
        seed=cfg.seed,
        work_types=getattr(cfg, "work_types", None),
        sort=getattr(cfg, "sort", None),
        stratify_by_year=getattr(cfg, "stratify_by_year", False),
        use_random_sample=getattr(cfg, "use_random_sample", False),
    )


@app.command()
def prepare(
    config: Path = typer.Option(..., "--config", "-c", path_type=Path, help="Data config YAML"),
) -> None:
    """Normalize raw JSONL to parquet in data/processed."""
    root = _ensure_project_cwd()
    cfg = load_data_config(root / config)
    raw_path = root / cfg.output_path
    if not raw_path.exists():
        typer.echo(f"Raw file not found: {raw_path}. Run fetch first.")
        raise typer.Exit(1)
    records = load_jsonl(raw_path)
    df = normalize_works_to_dataframe(records)
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{cfg.dataset_name}.parquet"
    write_parquet(df, out_path)
    logger.info("Wrote %s", out_path)


@app.command()
def validate(
    processed_path: Path = typer.Option(
        None, "--processed-path", path_type=Path, help="Path to processed parquet"
    ),
    data_config: Path = typer.Option(
        None, "--data-config", path_type=Path, help="Data config (used to resolve processed path)"
    ),
) -> None:
    """Run dataset validation on processed data; save report to artifacts/reports. Fail if validation fails."""
    root = _ensure_project_cwd()
    if processed_path is None and data_config is None:
        typer.echo("Provide either --processed-path or --data-config.", err=True)
        raise typer.Exit(1)
    expected_work_types: list[str] | None = None
    if processed_path is None:
        cfg = load_data_config(root / data_config)
        processed_path = root / "data" / "processed" / f"{cfg.dataset_name}.parquet"
        expected_work_types = getattr(cfg, "work_types", None)
    else:
        processed_path = root / processed_path
        if data_config is not None:
            cfg = load_data_config(root / data_config)
            expected_work_types = getattr(cfg, "work_types", None)
    if not processed_path.exists():
        typer.echo(f"Processed file not found: {processed_path}", err=True)
        raise typer.Exit(1)
    df = read_parquet(processed_path)
    run_id = processed_path.stem
    dataset_mode = "representative" if "representative" in run_id else ("temporal" if "temporal" in run_id else None)
    reports_dir = root / "artifacts" / "reports"
    from scholarly_outcome_prediction.validation.dataset_validation import (
        run_validation_and_save,
        DEFAULT_MIN_ROW_COUNT,
        DEFAULT_MIN_YEARS_WITH_DATA,
        DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
    )
    result, json_path, md_path = run_validation_and_save(
        raw_records=None,
        df=df,
        processed_path=processed_path,
        out_dir=reports_dir,
        run_id=None,
        min_row_count=DEFAULT_MIN_ROW_COUNT,
        min_years_with_data=DEFAULT_MIN_YEARS_WITH_DATA,
        max_venue_missingness_pct=DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
        dataset_mode=dataset_mode,
        expected_work_types=expected_work_types,
    )
    typer.echo(f"Validation report: {json_path}")
    if not result.get("passed", True):
        typer.echo("Validation failed: " + "; ".join(result.get("errors", [])), err=True)
        raise typer.Exit(1)
    typer.echo("Validation passed.")


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", path_type=Path, help="Experiment config YAML"
    ),
) -> None:
    """Train model and save pipeline to artifacts/models."""
    root = _ensure_project_cwd()
    cfg = load_experiment_config(root / config)
    set_global_seed(cfg.split.random_state)

    processed_path = root / cfg.data.processed_path
    if not processed_path.exists():
        typer.echo(f"Processed data not found: {processed_path}. Run prepare first.")
        raise typer.Exit(1)

    df = read_parquet(processed_path)
    num_feat = cfg.features.numeric
    cat_feat = cfg.features.categorical
    X, y = build_feature_matrix(
        df,
        numeric_features=num_feat,
        categorical_features=cat_feat,
        target_name=cfg.target.name,
        target_transform=cfg.target.transform or None,
    )
    full = pd.concat([X, y], axis=1)
    full = full.dropna(subset=[cfg.target.name])
    train_df, _ = train_test_split_df(
        full,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        split_kind=getattr(cfg.split, "split_kind", "random"),
        time_column=getattr(cfg.split, "time_column", None),
        train_year_end=getattr(cfg.split, "train_year_end", None),
        test_year_start=getattr(cfg.split, "test_year_start", None),
    )
    X_train = train_df[num_feat + cat_feat]
    y_train = train_df[cfg.target.name]

    preprocessor = build_preprocessor(num_feat, cat_feat)
    model_builder = get_model_builder(cfg.model.name)
    model = model_builder(params=cfg.model.params)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    art_dir = root / "artifacts" / "models"
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / f"{cfg.experiment_name}.joblib"
    save_model_pipeline(pipe, out_path)
    logger.info("Saved pipeline to %s", out_path)


@app.command()
def run(
    data_config: Path = typer.Option(..., "--data-config", path_type=Path, help="Data config YAML"),
    baseline_config: Path = typer.Option(
        ..., "--baseline-config", path_type=Path, help="Baseline experiment config"
    ),
    xgb_config: Path = typer.Option(
        ..., "--xgb-config", path_type=Path, help="XGBoost experiment config"
    ),
) -> None:
    """Run full pipeline: fetch -> prepare -> train (baseline + XGBoost) -> evaluate both."""
    root = _ensure_project_cwd()
    run_pipeline_from_configs(
        data_config_path=root / data_config,
        baseline_config_path=root / baseline_config,
        xgb_config_path=root / xgb_config,
    )


@app.command()
def evaluate(
    config: Path = typer.Option(
        ..., "--config", "-c", path_type=Path, help="Experiment config YAML"
    ),
) -> None:
    """Load model and data, compute metrics, save to artifacts/metrics."""
    root = _ensure_project_cwd()
    cfg = load_experiment_config(root / config)
    set_global_seed(cfg.split.random_state)

    model_path = root / "artifacts" / "models" / f"{cfg.experiment_name}.joblib"
    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}. Run train first.")
        raise typer.Exit(1)
    pipe = load_model_pipeline(model_path)

    processed_path = root / cfg.data.processed_path
    df = read_parquet(processed_path)
    num_feat = cfg.features.numeric
    cat_feat = cfg.features.categorical
    X, y = build_feature_matrix(
        df,
        numeric_features=num_feat,
        categorical_features=cat_feat,
        target_name=cfg.target.name,
        target_transform=cfg.target.transform or None,
    )
    full = pd.concat([X, y], axis=1)
    full = full.dropna(subset=[cfg.target.name])
    _, test_df = train_test_split_df(
        full,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        split_kind=getattr(cfg.split, "split_kind", "random"),
        time_column=getattr(cfg.split, "time_column", None),
        train_year_end=getattr(cfg.split, "train_year_end", None),
        test_year_start=getattr(cfg.split, "test_year_start", None),
    )
    X_test = test_df[num_feat + cat_feat]
    y_test = test_df[cfg.target.name].values

    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, metric_names=cfg.evaluation.metrics)
    effective_dataset_id = processed_path.stem
    _dataset_mode = "representative" if "representative" in effective_dataset_id else ("temporal" if "temporal" in effective_dataset_id else None)
    run_meta = build_run_metadata(
        experiment_name=cfg.experiment_name,
        target_name=cfg.target.name,
        target_transform=cfg.target.transform,
        target_mode=cfg.target.target_mode,
        model_name=cfg.model.name,
        model_params=cfg.model.params,
        feature_numeric=num_feat,
        feature_categorical=cat_feat,
        split_kind=getattr(cfg.split, "split_kind", "random"),
        split_test_size=cfg.split.test_size,
        split_random_state=cfg.split.random_state,
        train_size=len(full) - len(test_df),
        test_size=len(test_df),
        dataset_id=cfg.data.dataset_id,
        effective_dataset_id=effective_dataset_id,
        effective_processed_path=str(processed_path),
        dataset_mode=_dataset_mode,
    )
    out_path = root / "artifacts" / "metrics" / f"{cfg.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out_path, run_metadata=run_meta)
    logger.info("Saved metrics to %s", out_path)
    typer.echo(metrics)


def run_pipeline_from_configs(
    data_config_path: Path,
    baseline_config_path: Path,
    xgb_config_path: Path,
) -> None:
    """Run full pipeline: fetch -> prepare -> train baseline -> train xgb -> evaluate both."""
    setup_logging()
    # Project root: configs/data/foo.yaml -> parents[2] = project root
    root = data_config_path.resolve().parents[2]
    data_cfg = load_data_config(data_config_path)
    base_cfg = load_experiment_config(baseline_config_path)
    xgb_cfg = load_experiment_config(xgb_config_path)

    # Fetch
    out_path = root / data_cfg.output_path
    fetch_and_save(
        output_path=out_path,
        sample_size=data_cfg.sample_size,
        from_publication_date=data_cfg.from_publication_date,
        to_publication_date=data_cfg.to_publication_date,
        seed=data_cfg.seed,
        work_types=getattr(data_cfg, "work_types", None),
        sort=getattr(data_cfg, "sort", None),
        stratify_by_year=getattr(data_cfg, "stratify_by_year", False),
        use_random_sample=getattr(data_cfg, "use_random_sample", False),
    )
    # Prepare
    records = load_jsonl(out_path)
    if not records:
        typer.echo(
            "Error: fetch returned 0 works. Check data config (work_types, date range) and OpenAlex API. "
            "OpenAlex uses type:article for journal/proceedings articles, not journal-article.",
            err=True,
        )
        raise typer.Exit(1)
    df = normalize_works_to_dataframe(records)
    processed_path = root / "data" / "processed" / f"{data_cfg.dataset_name}.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, processed_path)

    # Validation (dataset_mode from config name for representative vs temporal thresholds)
    from datetime import datetime, timezone

    run_instance_id = datetime.now(timezone.utc).isoformat()
    reports_dir = root / "artifacts" / "reports"
    dataset_id = data_cfg.dataset_name
    dataset_mode = "representative" if "representative" in dataset_id else ("temporal" if "temporal" in dataset_id else None)
    from scholarly_outcome_prediction.validation.dataset_validation import (
        DEFAULT_MIN_ROW_COUNT,
        DEFAULT_MIN_YEARS_WITH_DATA,
        DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
    )
    try:
        from_date = data_cfg.from_publication_date
        to_date = data_cfg.to_publication_date
        expected_year_min = int(from_date[:4]) if from_date else None
        expected_year_max = int(to_date[:4]) if to_date else None
    except (ValueError, TypeError):
        expected_year_min = expected_year_max = None
    expected_work_types = getattr(data_cfg, "work_types", None)
    validation_result, validation_json_path, _ = run_validation_and_save(
        raw_records=records,
        df=df,
        processed_path=processed_path,
        out_dir=reports_dir,
        run_id=run_instance_id,
        min_row_count=DEFAULT_MIN_ROW_COUNT,
        min_years_with_data=DEFAULT_MIN_YEARS_WITH_DATA,
        max_venue_missingness_pct=DEFAULT_MAX_VENUE_MISSINGNESS_PCT,
        expected_year_min=expected_year_min,
        expected_year_max=expected_year_max,
        dataset_mode=dataset_mode,
        expected_work_types=expected_work_types,
    )
    if not validation_result.get("passed", True):
        typer.echo("Validation failed: " + "; ".join(validation_result.get("errors", [])), err=True)
        raise typer.Exit(1)
    logger.info("Validation passed; report at %s", validation_json_path)

    for cfg in (base_cfg, xgb_cfg):
        set_global_seed(cfg.split.random_state)
        full_df = read_parquet(processed_path)
        num_feat = cfg.features.numeric
        cat_feat = cfg.features.categorical
        eligibility_info: dict = {}
        if getattr(cfg.target, "target_mode", None) == "calendar_horizon":
            from scholarly_outcome_prediction.features.targets import prepare_df_for_target
            full_df, eligibility_info = prepare_df_for_target(
                full_df,
                target_name=cfg.target.name,
                target_mode=cfg.target.target_mode,
                horizon_years=getattr(cfg.target, "horizon_years", None),
                include_publication_year=getattr(cfg.target, "include_publication_year", True),
            )
        X, y = build_feature_matrix(
            full_df,
            numeric_features=num_feat,
            categorical_features=cat_feat,
            target_name=cfg.target.name,
            target_transform=cfg.target.transform or None,
        )
        full = pd.concat([X, y], axis=1)
        full = full.dropna(subset=[cfg.target.name])
        train_df, test_df = train_test_split_df(
            full,
            test_size=cfg.split.test_size,
            random_state=cfg.split.random_state,
            split_kind=getattr(cfg.split, "split_kind", "random"),
            time_column=getattr(cfg.split, "time_column", None),
            train_year_end=getattr(cfg.split, "train_year_end", None),
            test_year_start=getattr(cfg.split, "test_year_start", None),
        )
        X_train = train_df[num_feat + cat_feat]
        y_train = train_df[cfg.target.name]
        X_test = test_df[num_feat + cat_feat]
        y_test = test_df[cfg.target.name].values

        preprocessor = build_preprocessor(num_feat, cat_feat)
        model = get_model_builder(cfg.model.name)(params=cfg.model.params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        model_dir = root / "artifacts" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_model_pipeline(pipe, model_dir / f"{cfg.experiment_name}.joblib")

        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, metric_names=cfg.evaluation.metrics)
        experiment_config_path_used = (
            baseline_config_path if cfg.experiment_name == base_cfg.experiment_name else xgb_config_path
        )
        run_meta = build_run_metadata(
            experiment_name=cfg.experiment_name,
            target_name=cfg.target.name,
            target_transform=cfg.target.transform,
            target_mode=cfg.target.target_mode,
            model_name=cfg.model.name,
            model_params=cfg.model.params,
            feature_numeric=num_feat,
            feature_categorical=cat_feat,
            split_kind=getattr(cfg.split, "split_kind", "random"),
            split_test_size=cfg.split.test_size,
            split_random_state=cfg.split.random_state,
            train_size=len(train_df),
            test_size=len(test_df),
            dataset_id=cfg.data.dataset_id,
            effective_dataset_id=dataset_id,
            effective_processed_path=str(processed_path),
            data_config_path=str(data_config_path),
            experiment_config_path=str(experiment_config_path_used),
            validation_summary_path=str(validation_json_path),
            train_year_end=getattr(cfg.split, "train_year_end", None),
            test_year_start=getattr(cfg.split, "test_year_start", None),
            dataset_mode=dataset_mode,
            target_source=getattr(cfg.target, "source", None),
            horizon_years=getattr(cfg.target, "horizon_years", None),
            include_publication_year=getattr(cfg.target, "include_publication_year", None),
            target_eligibility=eligibility_info if eligibility_info else None,
        )
        meta_dir = root / "artifacts" / "metrics"
        meta_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, meta_dir / f"{cfg.experiment_name}.json", run_metadata=run_meta)
        logger.info("%s metrics: %s", cfg.experiment_name, metrics)

    # Run-scoped pipeline trace
    diag_dir = root / "artifacts" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    metrics_paths = [meta_dir / f"{c.experiment_name}.json" for c in (base_cfg, xgb_cfg)]
    model_paths = [root / "artifacts" / "models" / f"{c.experiment_name}.joblib" for c in (base_cfg, xgb_cfg)]
    from scholarly_outcome_prediction.diagnostics.pipeline_trace import build_pipeline_trace_from_run_context
    trace = build_pipeline_trace_from_run_context(
        run_id=run_instance_id,
        data_config_path=data_config_path,
        data_cfg=data_cfg,
        baseline_config_path=baseline_config_path,
        base_cfg=base_cfg,
        xgb_config_path=xgb_config_path,
        xgb_cfg=xgb_cfg,
        effective_processed_path=processed_path,
        validation_json_path=validation_json_path,
        stages_completed={
            "fetch": True,
            "prepare": True,
            "validation": True,
            "train": True,
            "evaluate": True,
        },
        metrics_paths=metrics_paths,
        model_paths=model_paths,
        dataset_id=dataset_id,
    )
    from scholarly_outcome_prediction.utils.io import save_json
    save_json(trace, diag_dir / "pipeline_trace.json")
    logger.info("Pipeline trace written to %s", diag_dir / "pipeline_trace.json")

    # Generate remaining diagnostics (profile, artifact audit, design trace, etc.) alongside run-scoped trace
    from scholarly_outcome_prediction.diagnostics.generate_all import generate_all_diagnostics
    generate_all_diagnostics(
        root,
        processed_path,
        dataset_id=dataset_id,
        out_dir=diag_dir,
        configs_dir=root / "configs",
        artifacts_root=root / "artifacts",
        include_design_trace=True,
    )
    logger.info("Diagnostics written to %s (pipeline_trace.json + profile, audit, design trace, etc.)", diag_dir)


if __name__ == "__main__":
    app()
