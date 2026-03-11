"""CLI: fetch, prepare, train, evaluate, run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
import typer

from scholarly_outcome_prediction.data import normalize_works_to_dataframe, train_test_split_df
from scholarly_outcome_prediction.evaluation import (
    build_run_metadata,
    compute_calibration_tail_metrics,
    compute_metrics,
    compute_zero_inflation_metrics,
    load_model_pipeline,
    save_metrics,
    save_model_pipeline,
)
from scholarly_outcome_prediction.features import build_feature_matrix, build_preprocessor
from scholarly_outcome_prediction.logging_utils import setup_logging, get_logger
from scholarly_outcome_prediction.models import get_model_builder
from scholarly_outcome_prediction.settings import load_data_config, load_current_experiment_config
from scholarly_outcome_prediction.utils.io import load_jsonl, read_parquet, write_parquet
from scholarly_outcome_prediction.utils.seeds import set_global_seed

from scholarly_outcome_prediction.acquisition import fetch_and_save
from scholarly_outcome_prediction.validation import run_validation_and_save
from scholarly_outcome_prediction.evaluation.benchmark_analysis import run_benchmark_analysis
from scholarly_outcome_prediction.diagnostics import run_overlap_audit

app = typer.Typer(help="Scholarly outcome prediction pipeline")
logger = get_logger(__name__)


def _ensure_project_cwd() -> Path:
    """Use current dir as project root (config paths are relative)."""
    return Path.cwd()


def _ensure_time_column_for_split(
    full: pd.DataFrame,
    source_df: pd.DataFrame,
    split_cfg: object,
) -> None:
    """When split is time-based and the time column is not in full (e.g. ablated), add it from source_df."""
    if getattr(split_cfg, "split_kind", "random") != "time":
        return
    time_col = getattr(split_cfg, "time_column", None)
    if not time_col or time_col in full.columns:
        return
    if time_col not in source_df.columns:
        return
    full[time_col] = source_df.loc[full.index, time_col].values


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
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Bypass OpenAlex raw fetch cache and re-download from API",
    ),
) -> None:
    """Fetch OpenAlex sample and save raw JSONL. Uses deterministic cache when same request identity."""
    root = _ensure_project_cwd()
    cfg = load_data_config(root / config)
    out_path = root / cfg.output_path
    cache_root = root / "artifacts" / "cache"
    result = fetch_and_save(
        output_path=out_path,
        sample_size=cfg.sample_size,
        from_publication_date=cfg.from_publication_date,
        to_publication_date=cfg.to_publication_date,
        seed=cfg.seed,
        work_types=getattr(cfg, "work_types", None),
        sort=getattr(cfg, "sort", None),
        stratify_by_year=getattr(cfg, "stratify_by_year", False),
        use_random_sample=getattr(cfg, "use_random_sample", False),
        force_refresh=force_refresh,
        cache_root=cache_root,
    )
    if result.from_cache:
        logger.info("Raw fetch reused from cache (key=%s); %d rows", result.cache_key, result.row_count)


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
    cfg = load_data_config(root / data_config) if data_config is not None else None
    if processed_path is None and cfg is not None:
        processed_path = root / "data" / "processed" / f"{cfg.dataset_name}.parquet"
    elif processed_path is not None:
        processed_path = root / processed_path
    if not processed_path.exists():
        typer.echo(f"Processed file not found: {processed_path}", err=True)
        raise typer.Exit(1)
    df = read_parquet(processed_path)
    expected_work_types = getattr(cfg, "work_types", None) if cfg is not None else None
    dataset_mode: str | None = None
    source_config_path: Path | None = None
    generation_params: dict | None = None
    if cfg is not None:
        dataset_mode = getattr(cfg, "dataset_mode", None) or (
            "representative" if "representative" in (cfg.dataset_name or "") else ("temporal" if "temporal" in (cfg.dataset_name or "") else None)
        )
        source_config_path = root / data_config
        generation_params = {
            "stratify_by_year": getattr(cfg, "stratify_by_year", False),
            "use_random_sample": getattr(cfg, "use_random_sample", False),
            "seed": getattr(cfg, "seed", 42),
            "sample_size": getattr(cfg, "sample_size", 0),
            "from_publication_date": getattr(cfg, "from_publication_date", ""),
            "to_publication_date": getattr(cfg, "to_publication_date", ""),
            "work_types": getattr(cfg, "work_types", None),
        }
    else:
        dataset_mode = "representative" if "representative" in processed_path.stem else ("temporal" if "temporal" in processed_path.stem else None)
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
        source_config_path=str(source_config_path) if source_config_path else None,
        generation_params=generation_params,
    )
    typer.echo(f"Validation report: {json_path}")
    if not result.get("passed", True):
        typer.echo("Validation failed: " + "; ".join(result.get("errors", [])), err=True)
        raise typer.Exit(1)
    typer.echo("Validation passed.")


@app.command("audit-dataset-overlap")
def audit_dataset_overlap(
    left: Path = typer.Option(..., "--left", "-l", path_type=Path, help="Path to first processed parquet"),
    right: Path = typer.Option(..., "--right", "-r", path_type=Path, help="Path to second processed parquet"),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        path_type=Path,
        help="Output directory for JSON and MD reports (default: artifacts/diagnostics)",
    ),
    id_column: str = typer.Option("openalex_id", "--id-column", help="Column used as work ID"),
    label_left: str | None = typer.Option(None, "--label-left", help="Label for left dataset in report"),
    label_right: str | None = typer.Option(None, "--label-right", help="Label for right dataset in report"),
) -> None:
    """Compare two processed datasets by work ID; report overlap and sample IDs. Use for representative vs temporal audit."""
    root = _ensure_project_cwd()
    left_path = left if left.is_absolute() else root / left
    right_path = right if right.is_absolute() else root / right
    if not left_path.exists():
        typer.echo(f"Left path not found: {left_path}", err=True)
        raise typer.Exit(1)
    if not right_path.exists():
        typer.echo(f"Right path not found: {right_path}", err=True)
        raise typer.Exit(1)
    out = out_dir if out_dir is not None else root / "artifacts" / "diagnostics"
    out = out if out.is_absolute() else root / out
    report, json_path, md_path = run_overlap_audit(
        path_left=left_path,
        path_right=right_path,
        out_dir=out,
        id_column=id_column,
        label_left=label_left,
        label_right=label_right,
    )
    if report.get("error"):
        typer.echo(report["error"], err=True)
        raise typer.Exit(1)
    typer.echo(f"Overlap report: {json_path}")
    typer.echo(f"Summary: {report['size_left']} vs {report['size_right']} rows; overlap={report['overlap_count']} ({report['overlap_rate_from_left_pct']}% from left); identical={report['identical']}")


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", path_type=Path, help="Experiment config YAML"
    ),
) -> None:
    """Train model and save pipeline to artifacts/models."""
    root = _ensure_project_cwd()
    try:
        cfg = load_current_experiment_config(root / config)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    set_global_seed(cfg.split.random_state)

    processed_path = root / cfg.data.processed_path
    if not processed_path.exists():
        typer.echo(f"Processed data not found: {processed_path}. Run prepare first.")
        raise typer.Exit(1)

    df = read_parquet(processed_path)
    if getattr(cfg.target, "target_mode", None) == "calendar_horizon":
        from scholarly_outcome_prediction.features.targets import prepare_df_for_target
        df, _ = prepare_df_for_target(
            df,
            target_name=cfg.target.name,
            target_mode=cfg.target.target_mode,
            horizon_years=getattr(cfg.target, "horizon_years", None),
            include_publication_year=getattr(cfg.target, "include_publication_year", True),
        )
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
    _ensure_time_column_for_split(full, df, cfg.split)
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
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Bypass OpenAlex raw fetch cache and re-download from API",
    ),
) -> None:
    """Run full pipeline: fetch -> prepare -> train (baseline + XGBoost) -> evaluate both."""
    root = _ensure_project_cwd()
    run_pipeline_from_configs(
        data_config_path=root / data_config,
        baseline_config_path=root / baseline_config,
        xgb_config_path=root / xgb_config,
        force_refresh=force_refresh,
    )


@app.command()
def evaluate(
    config: Path = typer.Option(
        ..., "--config", "-c", path_type=Path, help="Experiment config YAML"
    ),
) -> None:
    """Load model and data, compute metrics, save to artifacts/metrics."""
    root = _ensure_project_cwd()
    try:
        cfg = load_current_experiment_config(root / config)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    set_global_seed(cfg.split.random_state)

    model_path = root / "artifacts" / "models" / f"{cfg.experiment_name}.joblib"
    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}. Run train first.")
        raise typer.Exit(1)
    pipe = load_model_pipeline(model_path)

    processed_path = root / cfg.data.processed_path
    df = read_parquet(processed_path)
    if getattr(cfg.target, "target_mode", None) == "calendar_horizon":
        from scholarly_outcome_prediction.features.targets import prepare_df_for_target
        df, _ = prepare_df_for_target(
            df,
            target_name=cfg.target.name,
            target_mode=cfg.target.target_mode,
            horizon_years=getattr(cfg.target, "horizon_years", None),
            include_publication_year=getattr(cfg.target, "include_publication_year", True),
        )
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
    _ensure_time_column_for_split(full, df, cfg.split)
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
    metrics["zero_inflation"] = compute_zero_inflation_metrics(y_test, y_pred)
    metrics["calibration_tail"] = compute_calibration_tail_metrics(y_test, y_pred)
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
        train_year_end=getattr(cfg.split, "train_year_end", None),
        test_year_start=getattr(cfg.split, "test_year_start", None),
        benchmark_mode=getattr(cfg.benchmark, "benchmark_mode", None) if getattr(cfg, "benchmark", None) else None,
        model_family=getattr(cfg.benchmark, "model_family", None) if getattr(cfg, "benchmark", None) else None,
        is_diagnostic_model=getattr(cfg.benchmark, "is_diagnostic_model", None) if getattr(cfg, "benchmark", None) else None,
        ablation_name=cfg.ablation.name if getattr(cfg, "ablation", None) else None,
        ablation_features_removed=cfg.ablation.features_removed if getattr(cfg, "ablation", None) else None,
        ablation_type=getattr(cfg.ablation, "ablation_type", None) if getattr(cfg, "ablation", None) else None,
    )
    out_path = root / "artifacts" / "metrics" / f"{cfg.experiment_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out_path, run_metadata=run_meta)
    logger.info("Saved metrics to %s", out_path)
    logger.info(
        "metrics: rmse=%.4f mae=%.4f r2=%.4f",
        metrics.get("rmse"),
        metrics.get("mae"),
        metrics.get("r2"),
    )
    logger.debug("Full metrics: %s", metrics)


def run_pipeline_from_configs(
    data_config_path: Path,
    baseline_config_path: Path,
    xgb_config_path: Path,
    force_refresh: bool = False,
) -> None:
    """Run full pipeline: fetch -> prepare -> train baseline -> train xgb -> evaluate both."""
    setup_logging()
    # Project root: configs/data/foo.yaml -> parents[2] = project root
    root = data_config_path.resolve().parents[2]
    data_cfg = load_data_config(data_config_path)
    try:
        base_cfg = load_current_experiment_config(baseline_config_path)
        xgb_cfg = load_current_experiment_config(xgb_config_path)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    # Fetch (with deterministic cache; --force-refresh bypasses)
    out_path = root / data_cfg.output_path
    cache_root = root / "artifacts" / "cache"
    fetch_result = fetch_and_save(
        output_path=out_path,
        sample_size=data_cfg.sample_size,
        from_publication_date=data_cfg.from_publication_date,
        to_publication_date=data_cfg.to_publication_date,
        seed=data_cfg.seed,
        work_types=getattr(data_cfg, "work_types", None),
        sort=getattr(data_cfg, "sort", None),
        stratify_by_year=getattr(data_cfg, "stratify_by_year", False),
        use_random_sample=getattr(data_cfg, "use_random_sample", False),
        force_refresh=force_refresh,
        cache_root=cache_root,
    )
    if fetch_result.from_cache:
        logger.info("Raw fetch reused from cache (key=%s); %d rows", fetch_result.cache_key, fetch_result.row_count)
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

    # Validation (dataset_mode and provenance from data config)
    from datetime import datetime, timezone

    run_instance_id = datetime.now(timezone.utc).isoformat()
    reports_dir = root / "artifacts" / "reports"
    dataset_id = data_cfg.dataset_name
    dataset_mode = getattr(data_cfg, "dataset_mode", None) or (
        "representative" if "representative" in (dataset_id or "") else ("temporal" if "temporal" in (dataset_id or "") else None)
    )
    generation_params = {
        "stratify_by_year": getattr(data_cfg, "stratify_by_year", False),
        "use_random_sample": getattr(data_cfg, "use_random_sample", False),
        "seed": getattr(data_cfg, "seed", 42),
        "sample_size": getattr(data_cfg, "sample_size", 0),
        "from_publication_date": getattr(data_cfg, "from_publication_date", ""),
        "to_publication_date": getattr(data_cfg, "to_publication_date", ""),
        "work_types": getattr(data_cfg, "work_types", None),
    }
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
        source_config_path=str(data_config_path),
        generation_params=generation_params,
        raw_fetch_from_cache=fetch_result.from_cache,
        openalex_cache_key=fetch_result.cache_key,
        openalex_cache_path=fetch_result.cache_path,
    )
    if not validation_result.get("passed", True):
        typer.echo("Validation failed: " + "; ".join(validation_result.get("errors", [])), err=True)
        raise typer.Exit(1)
    logger.info("Validation passed; report at %s", validation_json_path)

    diag_dir = root / "artifacts" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    target_profile_path = None
    last_eligibility_info = None

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
            last_eligibility_info = eligibility_info
        X, y = build_feature_matrix(
            full_df,
            numeric_features=num_feat,
            categorical_features=cat_feat,
            target_name=cfg.target.name,
            target_transform=cfg.target.transform or None,
        )
        full = pd.concat([X, y], axis=1)
        full = full.dropna(subset=[cfg.target.name])
        # Generate target-level profile once per run for calendar_horizon (on first experiment)
        if getattr(cfg.target, "target_mode", None) == "calendar_horizon" and cfg.experiment_name == base_cfg.experiment_name:
            from scholarly_outcome_prediction.diagnostics.target_profile import (
                build_target_profile,
                write_target_profile,
                write_target_profile_md,
            )
            untransformed = full_df[cfg.target.name]
            profile = build_target_profile(
                eligibility_info,
                cfg.target,
                untransformed,
                transformed_target_series=y,
                run_id=run_instance_id,
                dataset_id=dataset_id,
                experiment_name=cfg.experiment_name,
                target_name=cfg.target.name,
            )
            target_profile_path = diag_dir / "target_profile.json"
            write_target_profile(profile, target_profile_path)
            write_target_profile_md(profile, diag_dir / "target_profile.md")
            logger.info("Target profile written to %s", target_profile_path)
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
        metrics["zero_inflation"] = compute_zero_inflation_metrics(y_test, y_pred)
        metrics["calibration_tail"] = compute_calibration_tail_metrics(y_test, y_pred)
        experiment_config_path_used = (
            baseline_config_path if cfg.experiment_name == base_cfg.experiment_name else xgb_config_path
        )
        target_semantics_description = None
        target_zero_rate = None
        if getattr(cfg.target, "target_mode", None) == "calendar_horizon":
            from scholarly_outcome_prediction.diagnostics.target_profile import build_target_semantics_description
            target_semantics_description = build_target_semantics_description(cfg.target, eligibility_info)
            ser = full_df[cfg.target.name].dropna()
            if len(ser):
                target_zero_rate = round(float((ser == 0).sum() / len(ser)), 4)
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
            target_semantics_description=target_semantics_description,
            target_zero_rate=target_zero_rate,
            benchmark_mode=getattr(cfg.benchmark, "benchmark_mode", None) if getattr(cfg, "benchmark", None) else None,
            model_family=getattr(cfg.benchmark, "model_family", None) if getattr(cfg, "benchmark", None) else None,
            is_diagnostic_model=getattr(cfg.benchmark, "is_diagnostic_model", None) if getattr(cfg, "benchmark", None) else None,
            ablation_name=cfg.ablation.name if getattr(cfg, "ablation", None) else None,
            ablation_features_removed=cfg.ablation.features_removed if getattr(cfg, "ablation", None) else None,
            ablation_type=getattr(cfg.ablation, "ablation_type", None) if getattr(cfg, "ablation", None) else None,
        )
        meta_dir = root / "artifacts" / "metrics"
        meta_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, meta_dir / f"{cfg.experiment_name}.json", run_metadata=run_meta)
        logger.info("%s metrics: %s", cfg.experiment_name, metrics)

    # Run-scoped pipeline trace
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
        target_profile_path=target_profile_path,
        target_eligibility_summary=last_eligibility_info,
        target_mode=getattr(base_cfg.target, "target_mode", None),
        target_source=getattr(base_cfg.target, "source", None),
        horizon_years=getattr(base_cfg.target, "horizon_years", None),
        include_publication_year=getattr(base_cfg.target, "include_publication_year", None),
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


@app.command(name="benchmark-analysis")
def benchmark_analysis(
    artifacts_dir: Path = typer.Option(
        None,
        "--artifacts-dir",
        path_type=Path,
        help="Artifacts root (default: ./artifacts)",
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        path_type=Path,
        help="Output directory for comparison/review (default: artifacts/reports)",
    ),
) -> None:
    """Generate unified benchmark comparison and ablation review from metrics artifacts."""
    root = _ensure_project_cwd()
    artifacts_root = artifacts_dir or root / "artifacts"
    summary = run_benchmark_analysis(artifacts_root, out_dir=out_dir)
    logger.info(
        "Benchmark analysis: loaded %s metrics, %s comparison rows, %s missing, %s ablations",
        summary["metrics_loaded"],
        summary["comparison_rows"],
        summary["comparison_missing"],
        summary["ablation_count"],
    )
    for label, path in summary["written"].items():
        logger.info("Wrote %s -> %s", label, path)
    logger.debug("Benchmark analysis summary: %s", summary)


if __name__ == "__main__":
    app()
